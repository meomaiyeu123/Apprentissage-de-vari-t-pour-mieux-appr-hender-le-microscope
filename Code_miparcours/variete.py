# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 17:20:14 2023

@author: LENOVO
"""
import os
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as img
from skimage.color import rgb2gray
from sklearn.manifold import LocallyLinearEmbedding
plt.style.use('default')

def load_image(file_name):
    PATH_NAME = os.getcwd()
    return iio.imread(PATH_NAME + "/" + file_name)

PSF = load_image("PSFs.tif")
(K,N,_) = PSF.shape
PSF_vector = np.array(PSF).reshape(K,N*N)
print("Le tenseur PSF est de taille",PSF.shape)
print("Il y a "+str(K)+" images de taille "+str(N)+" x "+str(N)) 

plt.subplot()
plt.title('Cellule à taille 42x42')
plt.imshow(PSF[0])
plt.show()


X = np.array(PSF).reshape(K, N*N)
k = 8
lle = LocallyLinearEmbedding(n_components = 2, n_neighbors=k)
X_reduced = lle.fit_transform(X)
fig = plt.figure(figsize = (8, 8))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title('LLE with k = '+str(k), size=12);

from matplotlib import offsetbox
selected_images = np.random.permutation(PSF_vector)


def plot_components(data, model, images=None, ax=None,
                    thumb_frac=0.05):
    ax = ax or plt.gca()
    
    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1],'.')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], zoom=1),
                                      proj[i])
            ax.add_artist(imagebox)
        for i in selected_images:
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i]), 
                proj[i])
            ax.add_artist(imagebox)
            
plt.figure(figsize=(8,8))
plot_components(X,
                model=LocallyLinearEmbedding(n_components=2, n_neighbors=k),
                thumb_frac=0.1,
                images=PSF);

