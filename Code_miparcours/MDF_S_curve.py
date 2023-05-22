# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:06:29 2023

@author: LENOVO
"""
from sklearn.datasets import make_s_curve
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
X, color = make_s_curve(1000, random_state=0)
plt.style.use('default')
#plt.rcParams['figure.facecolor'] = 'white'

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))

# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.jet);
plt.title('S-curve in $\mathbb{R}^3$')

# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
plt.title('Alternate view')
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.jet)
ax.view_init(4,90);

########################################

from sklearn.manifold import MDS
model = MDS(n_components=2, random_state=2)
out = model.fit_transform(X)
fig = plt.figure(figsize=(4,4))
plt.scatter(out[:, 0], out[:, 1], c=color, cmap=plt.cm.jet)
plt.title('2D MDS Projection')
plt.show();