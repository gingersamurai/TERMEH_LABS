import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=[7, 7])
ax = fig.add_subplot(1, 1, 1)
ax.set(xlim=[-20, 20],
       ylim=[-35, 5])

Steps = 1000
t = np.linspace(0, 10, Steps)
Phi = np.pi / 2 * np.cos(t)
Ksi = 1 + 2 * t

R = 3

Ox = 0
Oy = R
Bx = Ksi * np.sin(Phi) 
By = - Ksi * np.cos(Phi) + Oy

for i in range(10):
    print(f'phi: {Phi[i]}\t bx: {Bx[i]}\t by: {By[i]}\t{int(np.cos(Phi[i]))}\t{np.sin(Phi[i])} ')

beta = np.linspace(0, 2 * np.pi, 1000)
# beta = np.zeros_like(t)
# for i in range(len(beta)):
#     beta[i] = np.pi - Phi[i]
X_circle = R * np.sin(beta) + R
Y_circle = R * np.cos(beta)

Line = ax.plot([-4, 4], [Oy, Oy], 'g', lw=4)
Thread = ax.plot([Bx[0], Ox], [By[0], Oy], 'b', lw=4)[0]
Circle = ax.plot(X_circle + Bx[0], Y_circle + By[0], 'k', lw=2)[0]
Centre = ax.plot(Bx[0] + R, By[0], 'k', marker='o', ms=5)[0]
Base = ax.plot(Ox, Oy, 'r', marker='o', ms=10)[0]


def Animation(i):
    Circle.set_data(X_circle + Bx[i], Y_circle + By[i])
    Centre.set_data(Bx[i] + R, By[i])
    Thread.set_data([Bx[i], Ox], [By[i], Oy])
    return [Centre, Thread, Circle]


show = FuncAnimation(fig, Animation, frames=Steps, interval=10)

plt.show()
