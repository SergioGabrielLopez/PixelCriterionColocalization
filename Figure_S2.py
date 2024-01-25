# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:01:37 2024

@author: lopez
"""

# We import a few libraries.
import matplotlib.pyplot as plt
import pandas as pd

# We import an excel spreadsheet for the violin plot below.
data = pd.read_excel('C:/Users/lopez/Desktop/Paper Christine/Definitive/Revision 1/PCCAND candidate 3/PCCOR_violin.xlsx')

# We generate the figure. 
plt.figure(figsize=(8.3,6))
plt.violinplot(data,showmeans=True)
plt.ylim((-1,1))
plt.xlim((0.0,2.0))
plt.hlines(0,0.0,2.0,colors=['red'],linestyles='dashed')
plt.tick_params(axis='both', labelsize=22)
plt.xticks([])
plt.tight_layout()
plt.savefig('Figure_S2.tiff',dpi=300)
plt.ylabel('PCC',size=22)
plt.show()