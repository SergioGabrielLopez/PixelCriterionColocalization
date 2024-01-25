# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:15:12 2023

@author: lopez
"""

# We import useful libraries.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage import filters, io
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pingouin as pg


def generate_cyto(img1,img2,method='all',thresh1=None,thresh2=None):
    """This function calculates the probability density function for the cytofluorogram of images 1 and 2."""
    im1_flat = img1.flatten()
    im2_flat = img2.flatten()
    if method == 'all': # implements the PCCall pixel selection criterion.
        list_pixels = list(zip(im1_flat,im2_flat))[::2]
        x = [i[0] for i in list_pixels]
        y = [i[1] for i in list_pixels]
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        return x,y,z
    elif method =='or': # implements the PCCor pixel selection criterion.
        list_pixels = list(zip(im1_flat,im2_flat))
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


# We import the data for Tom40 & Tom40 (Note: the pixel size is 15 nm).
img1_tom40 = io.imread('C:/Users/lopez/Desktop/Paper Christine/Definitive/Revision 1/PCCOR candidate 6/real_data/real_data/Figure6_Yeast/Tom40_Tom40/128x128 sections/10_J1_594_128x128.tif', as_gray=True)
img2_tom40 = io.imread('C:/Users/lopez/Desktop/Paper Christine/Definitive/Revision 1/PCCOR candidate 6/real_data/real_data/Figure6_Yeast/Tom40_Tom40/128x128 sections/10_J1_640_128x128.tif', as_gray=True)

# We generate the RGB image for Tom40 & Tom40.
rgb_tom40_tom40 = generate_RGB(img1_tom40,img2_tom40)
    
# We calculate the PCC values for Tom40 & Tom40.
PCCall_tom40, PCCor_tom40, PCCand_tom40 = calculate_PCC(img1_tom40,img2_tom40)

# We import the data for Tom40 & Cbp3.
img1_cbp3 = io.imread('C:/Users/lopez/Desktop/Paper Christine/Definitive/Revision 1/PCCOR candidate 6/real_data/real_data/Figure6_Yeast/Tom40_Cbp3/128x128 sections/54_D1_594_128x128.tif', as_gray=True)
img2_cbp3 = io.imread('C:/Users/lopez/Desktop/Paper Christine/Definitive/Revision 1/PCCOR candidate 6/real_data/real_data/Figure6_Yeast/Tom40_Cbp3/128x128 sections/54_D1_640_128x128.tif', as_gray=True)

# We generate the RGB image for Tom40 & Cbp3.
rgb_tom40_cbp3 = generate_RGB(img1_cbp3,img2_cbp3)
    
# We calculate the PCC values for Tom40 & Cbp3.
PCCall_cbp3, PCCor_cbp3, PCCand_cbp3 = calculate_PCC(img1_cbp3,img2_cbp3)

# The values below were obtained by running the portion of the script above on the relevant datasets found at https://zenodo.org/records/4553856
PCCall_tom40_tom40 = [0.740947, 0.693186, 0.785362, 0.786380, 0.803684, 0.743271, 0.833317, 0.838253, 0.724037, 0.837866]
PCCor_tom40_tom40 = [0.296573, 0.060177, 0.218593, 0.300850, 0.401790, 0.147828, 0.568595, 0.498466, 0.059969, 0.381753]
PCCand_tom40_tom40 = [0.279262, 0.228668, 0.323272, 0.335703, 0.439612, 0.224287, 0.512801, 0.446864, 0.269827, 0.406372]

PCCall_tom40_cbp3 = [0.498034,0.205356,0.327465,0.417908,0.298717,0.319096,0.337205,0.388255,0.236041,0.377163]
PCCor_tom40_cbp3 = [-0.355813,-0.458600,-0.230245,-0.442931,-0.129639,-0.467490,-0.194999,-0.275212,-0.459459,-0.136444]
PCCand_tom40_cbp3 = [-0.073382,-0.457528,0.177610,-0.125555,0.018056,-0.294213,0.357492,0.071580,-0.444864,0.178049]

# We calculate one-sample and two-sample Student's t-tests and we print them out.
one_sample_t_stat_tom40_or, one_sample_p_value_tom40_or = stats.ttest_1samp(PCCor_tom40_tom40, 0)
one_sample_t_stat_cbp3_and, one_sample_p_value_cbp3_and = stats.ttest_1samp(PCCand_tom40_cbp3, 0)

t_stat_or, p_value_or = stats.ttest_ind(PCCor_tom40_tom40, PCCor_tom40_cbp3)
t_stat_and, p_value_and = stats.ttest_ind(PCCand_tom40_tom40, PCCand_tom40_cbp3)

print('The p values for the PCCor and PCCand criteria are {} and {}, respectively. PCCor is {} times smaller then PCCand.\n'.format(p_value_or,p_value_and,p_value_and/p_value_or))

# We calculate the effect size using Hedges' g test.
eff_size_PCC_and = pg.compute_effsize(PCCand_tom40_tom40, PCCand_tom40_cbp3, eftype='hedges') 
eff_size_PCC_or = pg.compute_effsize(PCCor_tom40_tom40, PCCor_tom40_cbp3, eftype='hedges') 

# Now we create a pandas dataframe from the lists above.
dict_PCCs = {'PCCall Tom40 & Tom40':PCCall_tom40_tom40,'PCCor Tom40 & Tom40':PCCor_tom40_tom40,'PCCand Tom40 & Tom40':PCCand_tom40_tom40,'PCCall Tom40 & Cbp3':PCCall_tom40_cbp3, 'PCCor Tom40 & Cbp3':PCCor_tom40_cbp3, 'PCCand Tom40 & Cbp3':PCCand_tom40_cbp3}
df = pd.DataFrame(dict_PCCs)
    
# We import an excel spreadsheet and create a few variables for the violin plot below.
dfswarm = pd.read_excel('C:/Users/lopez/Desktop/Paper Christine/Definitive/Revision 1/PCCOR candidate 6/swarmplot_data_2.xlsx')
columns_my_order = ['PCCor Tom40 & Tom40', 'PCCor Tom40 & Cbp3','PCCand Tom40 & Tom40','PCCand Tom40 & Cbp3']
positions = [1,1.1,1.3,1.4]
colors = ['green','red','green','red']

# We create the figure.
fig,ax = plt.subplots(2,2,figsize=(16.9,14))
ax[0,0].imshow(rgb_tom40_tom40)
ax[0,0].axis('off')
ax[0,0].text(115,123,'A',size=45,color='white')
ax[0,0].text(3,8,'$ PCC_{ALL} = $' + "%0.2f" % (PCCall_tom40), size=24, color='w')
ax[0,0].text(3,16,'$ PCC_{OR} = $' + "%0.2f" % (PCCor_tom40), size=24, color='w')
ax[0,0].text(3,24,'$ PCC_{AND} = $' + "%0.2f" % (PCCand_tom40), size=24, color='w')
ax[0,0].text(3,115,'Tom40 - SR', size=24, color='g')
ax[0,0].text(3,123,'Tom40 - AF594', size=24, color='magenta')
scalebar = AnchoredSizeBar(ax[0,0].transData, 33, label='', loc='upper right', pad=0.1, borderpad=1, color='white', frameon=False, size_vertical=3) # Scalebar is 500 nm.
ax[0,0].add_artist(scalebar)
ax[0,1].imshow(rgb_tom40_cbp3)
ax[0,1].axis('off')
ax[0,1].text(115,123,'B',size=45,color='white')
ax[0,1].text(3,8,'$ PCC_{ALL} = $' + "%0.2f" % (PCCall_cbp3), size=24, color='w')
ax[0,1].text(3,16,'$ PCC_{OR} = $' + "%0.2f" % (PCCor_cbp3), size=24, color='w')
ax[0,1].text(3,24,'$ PCC_{AND} = $' + "%0.2f" % (PCCand_cbp3), size=24, color='w')
ax[0,1].text(3,115,'Cbp3 - SR', size=24, color='g')
ax[0,1].text(3,123,'Tom40 - AF594', size=24, color='magenta')
scalebar = AnchoredSizeBar(ax[0,1].transData, 33, label='', loc='upper right', pad=0.1, borderpad=1, color='white', frameon=False, size_vertical=3)
ax[0,1].add_artist(scalebar)
for i in range(len(columns_my_order)):
    parts = ax[1,0].violinplot(df[columns_my_order[i]], positions=[positions[i]], showmeans=True, widths=0.1)
    parts['bodies'][0].set_color(colors[i])
    parts['cmeans'].set_colors(colors[i])
    parts['cbars'].set_colors(colors[i])
    parts['cmaxes'].set_colors(colors[i])
    parts['cmins'].set_colors(colors[i])
ax[1,0].set_xticklabels(columns_my_order, color='white')
ax[1,0].text(0.95,-0.78,'Tom40 & Tom40', size=24, color='green')
ax[1,0].text(0.95,-0.9,'Tom40 & Cbp3', size=24, color='red')
ax[1,0].text(1.01,0.8,'$ PCC_{OR}$', size=24, color='k')
ax[1,0].text(0.99,0.70,'p = ' + "%0.1e" % (p_value_or), size=18, color='k')
ax[1,0].text(1.32,0.8,'$ PCC_{AND}$', size=24, color='k')
ax[1,0].text(1.31,0.70,'p = ' + "%0.1e" % (p_value_and), size=18, color='k')
ax[1,0].tick_params(axis='both', labelsize=20)
ax[1,0].set_xticks([])
ax[1,0].set_ylim(-1.0,1.0)
ax[1,0].set_ylabel('PCC',size=20)
ax[1,0].text(1.43,-0.95,'C',size=45,color='black')
thresh_1 = filters.threshold_otsu(img1_tom40)
thresh_2 = filters.threshold_otsu(img2_tom40)
x, y, z = generate_cyto(img1_tom40,img2_tom40)
coef_all = np.polyfit(x,y,1)
poly1d_fn_all = np.poly1d(coef_all) 
kde_data_dict = {'x':list(img1_tom40.flatten()),'y':list(img2_tom40.flatten())}
kde_data = pd.DataFrame(kde_data_dict)
sns.kdeplot(data=kde_data, x="x", y="y", levels=10, clip=(0.0,1.0), cmap='hot', fill=True, cbar=True, ax=ax[1,1])
ax[1,1].plot(x, poly1d_fn_all(x), '--r',linewidth=1.5, dashes=(5,10))
ax[1,1].set_xlim(0.0,1.0)
ax[1,1].set_ylim(0.0,1.0)
ax[1,1].vlines(thresh_1,0.0,1.0)
ax[1,1].hlines(thresh_2,0.0,1.0)
ax[1,1].fill_between([thresh_1,1], [thresh_2, thresh_2],color='red',alpha=0.2)
ax[1,1].fill_between([0,thresh_1], [thresh_2, thresh_2], 1, color='blue',alpha=0.2)
ax[1,1].fill_between([thresh_1,1], [thresh_2, thresh_2], 1, color='green',alpha=0.2)
ax[1,1].text(0.03,0.85,'1',color='blue',size=28)
ax[1,1].text(0.8,0.85,'2',color='green',size=28)
ax[1,1].text(0.8,0.12,'3',color='red',size=28)
ax[1,1].set_xlabel('Tom40 - Alexa Fluor 594', size=20)
ax[1,1].set_ylabel('Tom40 - STAR RED', size=20)
ax[1,1].tick_params(axis='both', labelsize=20)
ax[1,1].text(0.90,0.02,'D',size=45,color='black')
cax = ax[1,1].figure.axes[-1]
cax.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig('Figure_4.tiff',dpi=300)
plt.show()
