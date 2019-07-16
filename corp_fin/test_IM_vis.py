# visualize import export

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
import pickle
import os
import matplotlib.animation as animation
import matplotlib.patches as patches

sys.path.insert(0,'C:/Users/tanel.joon/OneDrive - Energia.ee/Documents_OneDrive/Python/for_import')


from pyshp import shapefile
import numpy as np

folder  = 'C:/Users/tanel.joon/OneDrive - Energia.ee/Documents_OneDrive/Python/data/maps/'

#filename = 'gadm36_' + DNK + '_0.shp'

#states_list = ['DNK', 'DEU', 'GBR', 'NLD', 'NOR', 'SWE']
states_list = ['DNK', 'DEU', 'NLD']
main = 'DNK'

#plt.figure()

#fig, ax = plt.gca()
fig, ax = plt.subplots()
ax.set_aspect(1)

#arrow = ax.arrow(56.11, 9.34, 0, -10, color ='green', linewidth = 4)
patch = patches.FancyArrowPatch((9, 56), (9, 52), mutation_scale=100)
#patch2 = patches.FancyArrowPatch((10, 52), (10, 56), mutation_scale=100)

#ax.add_patch(arrow)

T = 50
t = np.arange(T)
dt = 0.05
time_template = 'time = %.1fs'
time_text = ax.text(10, 50, '')

def init():
    ax.add_patch(patch)
    ax.add_patch(patch)
    time_text.set_text('')
    
    return patch, time_text
    
def animate(t):
    global patch
    temp = random.random()

    ax.patches.remove(patch)
    
    patch = patches.FancyArrowPatch((9, 56), (9, 52),mutation_scale=100 * temp)
    ax.add_patch(patch)
    
    time_text.set_text(time_template % (t*dt))
    return patch, time_text
    
    
for ii in states_list:
    target = folder + 'gadm36_' + ii + '_0.shp'
    print(target)

    sf = shapefile.Reader(target)

    shape = sf.shape() # whichever shape
    points = np.array(shape.points)

    intervals = list(shape.parts) + [len(shape.points)]

    for (i, j) in zip(intervals[:-1], intervals[1:]):
        
        if ii == main:
            ax.fill(*zip(*points[i:j]), color ='blue')
            ax.plot(*zip(*points[i:j]), color ='black')
        else:
            ax.fill(*zip(*points[i:j]), color ='red')
            ax.plot(*zip(*points[i:j]), color ='black')
# import den

ani = animation.FuncAnimation(fig, animate, T, init_func=init,
                              interval=100, blit=True)

plt.show()