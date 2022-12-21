import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as pat
import sympy as sp
import math


t = sp.symbols('t')

phi = t
thetta = 2 * np.pi * sp.sin(t)

# 
OA = 5
AB = 4

X_A = -1 * OA * sp.sin(phi)
Y_A = OA * sp.cos(phi)

X_B = X_A + AB * sp.sin(thetta)
Y_B = Y_A - AB * sp.cos(thetta)

spr_t = sp.symbols('spr_t')
spr_phi = spr_t
spr_r = 0.05 * spr_t
X_spr = spr_r * sp.cos(spr_phi)
Y_spr = spr_r * sp.sin(spr_phi)

spr_T = np.linspace(0, 4 * np.pi, 100)
X_sprt = np.zeros_like(spr_T)
Y_sprt = np.zeros_like(spr_T)

for i in np.arange(len(spr_T)):
    X_sprt[i] = sp.Subs(X_spr, spr_t, spr_T[i])
    Y_sprt[i] = sp.Subs(Y_spr, spr_t, spr_T[i])

def Rod2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY

X_sprt, Y_sprt = Rod2D(X_sprt, Y_sprt, np.pi / 2)

# 
T = np.linspace(0, 100, 1000)

X_At = np.zeros_like(T)
Y_At = np.zeros_like(T)

X_Bt = np.zeros_like(T)
Y_Bt = np.zeros_like(T)

phi_t = np.zeros_like(T)
for i in np.arange(len(T)):
    X_At[i] = sp.Subs(X_A, t, T[i])
    Y_At[i] = sp.Subs(Y_A, t, T[i])

    X_Bt[i] = sp.Subs(X_B, t, T[i])
    Y_Bt[i] = sp.Subs(Y_B, t, T[i])

    phi_t[i] = sp.Subs(phi, t, T[i])



    


fig = plt.figure()

ax_main = fig.add_subplot(1, 1, 1)
ax_main.axis("equal")
ax_main.set(xlim=[-10, 10], ylim=[-10, 10])


ax_main.plot([0, -1, 1, 0], [0, -1, -1, 0], color='black') #triangle
ax_main.plot([-12, 12], [-1,-1], 'black', linestyle = '--') #line


pt_OA, = ax_main.plot([0, X_At[0]], [0, Y_At[0]], color="blue") #OA
pt_AB, = ax_main.plot([X_At[0], X_Bt[0]], [Y_At[0], Y_Bt[0]], color="blue") #AB

ax_main.plot(0, 0, marker='o', color="red") #O
pt_A, = ax_main.plot(X_At[1], Y_At[1], marker='o', color='red') #A
pt_B, = ax_main.plot(X_Bt[0], Y_Bt[0], marker='o', markersize=10, color='magenta') #B

pt_spr, = ax_main.plot(X_sprt, Y_sprt)

def anima(i):
    pt_A.set_data(X_At[i], Y_At[i])
    pt_B.set_data(X_Bt[i], Y_Bt[i])

    pt_OA.set_data([0, X_At[i]], [0, Y_At[i]])
    pt_AB.set_data([X_At[i], X_Bt[i]], [Y_At[i], Y_Bt[i]])

    new_spr_x, new_spr_y = Rod2D(X_sprt, Y_sprt, phi_t[i])
    pt_spr.set_data(new_spr_x, new_spr_y)


anim = FuncAnimation(fig, anima, frames=1000, interval=50, blit=False, repeat=True)
plt.show()