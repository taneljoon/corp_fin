# test time plot

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from time import sleep
import numpy as np
import weakref

import sys

dt = 0.1
t = np.arange(0.0, 50, dt)

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-50, 50), ylim=(-50, 50))
ax.grid()

line, = ax.plot([], [], 'ro', lw=2)

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def check_distance(point1, point2, distance):
    distance_check = ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
    if distance_check <= distance:
        out = True
    else:
        out = False
    return out

class Point(object):
    instances = []
    def __init__(self, name, x, y, ax):
        self.__class__.instances.append(weakref.proxy(self))
        self.name = name
        self.x = x
        self.y = y
        self.point, = ax.plot([], [],'ro')
    
    def first(self):
        self.point.set_data([], [])
        
        return self.point
    def calc(self):
        self.x = self.x + random.random() - random.random()
        self.y = self.y + random.random() - random.random()
        self.point.set_data(self.x, self.y)
        
        return self.point

point1 = Point('tank',1,1, ax)       
point2 = Point('chopper', 4,4, ax)     
  
# ________________________________________________
def init():
    out = []
    for ii in Point.instances:
        temp = ii.first()
        out.append(temp)
    
    time_text.set_text('')
    
    out.append(time_text)
    out_tuple = tuple(out)
    
    #return time_text
    return out_tuple

def animate(i):
    out = []
    for ii in Point.instances:
        temp = ii.calc()
        out.append(temp)
         
    time_text.set_text(time_template % (i*dt))
    sleep(dt)
    
    out.append(time_text)
    out_tuple = tuple(out)
    return out_tuple


ani = animation.FuncAnimation(fig, animate, np.arange(1, len(t)),
                              interval=25, blit=True, init_func=init)


plt.show()
