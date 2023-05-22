# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:30:18 2023

@author: NGUYEN Minh-Duy
"""
from sklearn.manifold import Isomap
import projet_lib

def load_image(file_name):
    PATH_NAME = os.getcwd()
    return iio.imread(PATH_NAME + "/" + file_name)

PSF = load_image("PSFs.tif")
(K,N,_) = PSF.shape
            
#ISOMAP
np.random.seed(0)
PSF_vector = np.array(PSF).reshape(K, N*N)
data = np.random.permutation(PSF_vector)
images = np.array(data).reshape(K,N,N)
fig, ax = plt.subplots(figsize=(10, 10))
plot_components(data, model=Isomap(n_components=2, n_neighbors=8), images=images, thumb_frac=0.1)
plt.savefig("img_Iso2.png")