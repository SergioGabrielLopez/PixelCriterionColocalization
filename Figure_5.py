# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:05:51 2023

@author: lopez
"""

# We import a few libraries.
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, io, transform
import pandas as pd
import scipy.stats as stats
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

def calculate_PCC(image1,image2):
    """Calculates the different PCC for the biological examples."""
    thresh1 = filters.threshold_otsu(image1)
    thresh2 = filters.threshold_otsu(image2)
    img1_mask = image1 > thresh1
    img2_mask = image2 > thresh2
    # The PCCall criterion.
    PCCall = calculate_Pearson(image1,image2)
    # The PCCor criterion.
    intersection_or = img1_mask | img2_mask
    PCCor = calculate_Pearson(image1[intersection_or],image2[intersection_or])
    # The PCCand criterion.
    intersection_and = img1_mask & img2_mask
    PCCand = calculate_Pearson(image1[intersection_and],image2[intersection_and])
    return PCCall, PCCor, PCCand

def calculate_Pearson(image1,image2):
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

# We import the images.
img1 = io.imread('C:/Users/lopez/Desktop/Paper Christine/Definitive/Revision 1/PCCAND candidate 3/Figure 5 - Colocalization/DSB1HAms_to_DSB3FLAGrb/Individual Nucleus Images and Peak Calls/A_023/C1.tif')
img2 = io.imread('C:/Users/lopez/Desktop/Paper Christine/Definitive/Revision 1/PCCAND candidate 3/Figure 5 - Colocalization/DSB1HAms_to_DSB3FLAGrb/Individual Nucleus Images and Peak Calls/A_023/C2.tif')

# We pre-process the images.
img1 = img1 - np.min(img1)
img1_rot = transform.rotate(img1,90) # we rotate this image by 90 degrees. 
img2 = img2 - np.min(img2)

# We calculate the PCC values.
PCCall, PCCor, PCCand = calculate_PCC(img1,img2)

# We calculate the PCC values for the rotated image.
PCCall_rot, PCCor_rot, PCCand_rot = calculate_PCC(img1_rot,img2)

# By running the script above on the 24 available images, we obtain the data below.
PCCand_list = [0.10763275,0.105959706,0.051624533,0.07643461,0.23452197,0.12264189,0.15423818,0.059690215,0.14659396,0.18945353,0.07013093,0.032324586,0.031884383,-0.029155415,-0.0033827762,0.059392806,0.14291893,0.08491543,-0.021272527,1.9275189e-05,0.007550109,0.18770848,0.24296032,0.0141166765]
PCCand_rot_list = [0.10685632,-0.09265367,0.025033828,0.017307768,0.078428075,-0.03996243,0.00050180295,-0.094684795,-0.057523534,0.07357851,-0.03497912,-0.008902503,0.016586035,0.0492466,0.058448493,-0.0061035845,-0.012393139,0.044885326,0.08077873,-0.08479949,0.014681767,0.1481246,-0.0251834,0.0076722875]

# We import an excel spreadsheet for the violin plot below.
data = pd.read_excel('C:/Users/lopez/Desktop/Paper Christine/Definitive/Revision 1/PCCAND candidate 3/data_violinplot_2.xlsx')
colors = ['green','red']
columns_my_order = ['PCCand','PCCand rot']
positions = [1.0,1.2]

# We calculate one-sample and two-sample Student's t-tests and we print them out.
one_sample_t, one_sample_p_value = stats.ttest_1samp(PCCand_list, 0)
one_sample_t_rot, one_sample_p_value_rot = stats.ttest_1samp(PCCand_rot_list, 0)
t_stat, p_value = stats.ttest_ind(PCCand_list, PCCand_rot_list) # Two saple Student's t-test.

# We generate the figure.
rgb = generate_RGB(img1[14,:,:]/np.max(img1[14,:,:]),img2[14,:,:]/np.max(img2[14,:,:]))
fig, ax = plt.subplots(1,2,figsize=(16.9,7))
ax[0].imshow(rgb**2.3)
ax[0].axis('off')
scalebar = AnchoredSizeBar(ax[0].transData, 50, label='', loc='lower right', pad=0.1, borderpad=1, color='white', frameon=False, size_vertical=3) # The scale bar is 2.0 microns
ax[0].add_artist(scalebar)
ax[0].text(163,20,'A',size=45,color='white')
ax[0].text(5,12,'$ PCC_{OR}$ = ' + "%0.2f" % (PCCor), size=24, color='w')
ax[0].text(5,25,'$ PCC_{AND}$ = ' + "%0.2f" % (PCCand), size=24, color='w')
ax[0].text(5,160,'DSB-1', size=28, color='magenta')
ax[0].text(5,175,'DSB-3', size=28, color='green')
for i in range(len(columns_my_order)):
    parts = ax[1].violinplot(data[columns_my_order[i]], positions=[positions[i]], showmeans=True, widths=0.1)
    parts['bodies'][0].set_color(colors[i])
    parts['cmeans'].set_colors(colors[i])
    parts['cbars'].set_colors(colors[i])
    parts['cmaxes'].set_colors(colors[i])
    parts['cmins'].set_colors(colors[i])
ax[1].tick_params(axis='both', labelsize=22)
ax[1].set_ylabel('PCC',size=22)
ax[1].text(1.23,0.22,'B',size=45,color='black')
ax[1].set_xticks([])
ax[1].text(1.05,0.22,'$ PCC_{AND}$', size=24, color='green')
ax[1].text(1.04,-0.06,'$ PCC_{AND}$ rotated', size=24, color='red')
ax[1].text(1.05,0.20,'p = ' + "%0.1e" % (one_sample_p_value), size=22, color='green')
ax[1].text(1.04,-0.08,'p = ' + "%0.1e" % (one_sample_p_value_rot), size=22, color='red')
plt.tight_layout()
plt.savefig('Figure_5.tiff',dpi=300)
plt.show()




