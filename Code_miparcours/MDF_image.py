# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:44:23 2023

@author: LENOVO
"""

from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import MDS 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import projet_lib

PSF = load_image("PSFs.tif")
(K,N,_) = PSF.shape

# get random subset of digits data
np.random.seed(0)
PSF_vector = np.array(PSF).reshape(K, N*N)
sample = np.random.permutation(PSF_vector)[:500]
labels = get_k_means_labels(sample)
fig = plt.figure(figsize=plt.figaspect(0.5))
D = pairwise_distances(sample)

# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1)
# plot results of MDS
X = ClassicalMDS(pairwise_distances(sample), 18)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.get_cmap('jet', 8))
plt.title('Classical MDS')

# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2)
clf = PCA(n_components=2)
X_pca = clf.fit_transform(sample)
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=plt.cm.get_cmap('jet', 8))
plt.title('PCA');
plt.savefig("img_MDS_PCA.png")

#MDS from the sklearn.manifold library
plt.figure(figsize=plt.figaspect(0.5))
model = MDS(n_components=2)
proj = model.fit_transform(sample)
plt.scatter(proj[:,0], proj[:,1], c=labels)
plt.title('MDS from manifold library');
plt.savefig("img_MDS_from_manifold_lib.png")

images = np.array(sample).reshape(500,N,N)
plot_components(proj, model=model, images=images,
                    thumb_frac=0.3)
