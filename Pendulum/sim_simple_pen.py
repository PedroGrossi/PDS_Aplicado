#%%
# d^2tht =  - (c/(m*l))*dtht - (g/l)*sin(tht[n-1])
# (tht[n] - 2tht[n-1] + tht[n-2])/dt**2 = - (c/(m*l))*(tht[n]-tht[n-2])/(2dt) - (dt**2*g/l)*sin(tht[n-1])
# (1 + (c*dt/(2*m*l)))*(tht[n]))= 2tht[n-1] + ((c*dt/(2*m*l)) - 1)*tht[n-2] - (dt**2*g/l)*sin(tht[n-1])
# tht[n] = a1*tht[n-1] + a2*tht[n-2] + a3*sin(tht[n-1])

import numpy as np
import matplotlib.pyplot as plt
from time import time

l = .5      # lenght [m]
m = 1       # mass [kg]
c = .075    # friction [N.ss/rad]
g = -9.81   # gravity [m/s^2]

Lt = 40     # Simulation time [s]
dt = 0.05   # Discretization [s]
Nt = round(Lt / dt)

tht0 = np.pi - 1e-9
dtht0 = 2*np.pi # angular speed [rad/s]
tht = tht0 * np.ones(Nt)
tht[0] = tht0
tht[1] = tht0 + dtht0*dt

a1 = 2 / (1 + dt*c / (2*m*l))
a2 = (dt*c / (2*m*l) - 1) / (1 + dt*c / (2*m*l))
a3 = dt**2*g / (l*(1 + dt*c / (2*m*l)))

fig, ax = plt.subplots()
ax.set_xlim(-l, l)
ax.set_ylim(-l, l)
line = ax.plot([0, l*np.sin(tht0)], [0, -l*np.cos(tht0)])

t2 = time()
for nt in range(2, Nt):
    t1 = t2
    tht[nt] = a1*tht[nt-1] + a2*tht[nt-2] + a3*np.sin(tht[nt-1])
    line[0].set_data([0, l*np.sin(tht[nt])], [0, -l*np.cos(tht[nt])])
    ax.set_title(f"t = {nt*dt:.2f}")
    t2 = time()
    delay = max(1e-12, dt - (t2 - t1))
    plt.pause(delay)

t = np.arange(Nt)*dt
plt.figure()
plt.plot(t, tht/(2*np.pi))
plt.show()