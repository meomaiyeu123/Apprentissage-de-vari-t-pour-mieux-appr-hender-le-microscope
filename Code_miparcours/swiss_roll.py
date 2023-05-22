# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:31:25 2023

@author: LENOVO
"""

import numpy as np
import matplotlib.pyplot as plt
# draw samples to create the grid
t = np.linspace(0, 1, 50)
u = np.linspace(0, 1, 50)
v = 3*np.pi/2*(.1 + 2*t)
u,v = np.meshgrid(u,v)

# swiss roll transformation
x = -v*np.cos(v)
y = u
z = v*np.sin(v)

fig = plt.figure(figsize=plt.figaspect(0.5))
colors = plt.cm.jet((x**2+z**2)/100)
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors, 
                cmap=plt.cm.coolwarm, linewidth=1.4, alpha=0.8)
ax.set_title('Continuous manifold in $\mathbb{R}^3$');

# draw uniform samples from the continuous manifold
n = 4000
t = np.random.rand(n, 1)
u = np.random.rand(n, 1)
v = 3*np.pi/2*(.1 + 2*t)

x = -v*np.cos(v)
y = u
z = v*np.sin(v)
color = (x**2 + z**2) / 100
color = color.reshape(n,)

# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(x, y, z, cmap='jet', c=color, marker='x');
ax.set_title('Finite Data Samples', size=16);
plt.show();