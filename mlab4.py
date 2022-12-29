import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as pat
import sympy as sp
from scipy.integrate import odeint

def odesys(y, t, g, l, c, k1, k2, m):
    # y = [phi, thetta, phi', thetta']
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = 1
    a12 = -np.cos(y[0] - y[1])
    a21 = -np.cos(y[0] - y[1])
    a22 = 1

    b1 = (g / l) * np.sin(y[0]) - (c * y[0] + k1 * y[2]) / (m * l**2) + y[3]**2 * np.sin(y[0] - y[1])
    b2 = -(g / l) * np.sin(y[1]) - k2 * y[3] / (m * l**2) - y[2]**2 * np.sin(y[0] - y[1])

    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    dy[3] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)

    return dy


m = 0.3
l = 5
c = 2
k1 = 1
k2 = 1
g = 9.81

t = sp.symbols('t')

phi0 = np.pi / 3
thetta0 = -np.pi / 3
dphi0 = 0
dthetta0 = 0


y0 = [phi0, thetta0, dphi0, dthetta0]

T = np.linspace(0, 100, 1000)


Y = odeint(odesys, y0, T, (g, l, c, k1, k2, m))

phi = Y[:, 0]
thetta = Y[:, 1]
dphi = Y[:, 2]
dthetta = Y[:, 3]

ddphi = [odesys(y,T, g, l, c, k1, k2, m)[2] for y, T in zip(Y, T)]
ddthetta = [odesys(y,T, g, l, c, k1, k2, m)[2] for y, T in zip(Y, T)]

ddphi = np.array(ddphi)
ddthetta = np.array(ddthetta)
# phi = t
# thetta = 2 * np.pi * sp.sin(t)

OA = 5
AB = 4

X_A = -1 * OA * np.sin(phi)
Y_A = OA * np.cos(phi)

X_B = X_A + AB * np.sin(thetta)
Y_B = Y_A - AB * np.cos(thetta)

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

X_At = X_A
Y_At = Y_A
X_Bt = X_B
Y_Bt = Y_B
phi_t = phi
# X_At = np.zeros_like(T)
# Y_At = np.zeros_like(T)

# X_Bt = np.zeros_like(T)
# Y_Bt = np.zeros_like(T)

# phi_t = np.zeros_like(T)
# for i in np.arange(len(T)):
#     X_At[i] = sp.Subs(X_A, t, T[i])
#     Y_At[i] = sp.Subs(Y_A, t, T[i])

#     X_Bt[i] = sp.Subs(X_B, t, T[i])
#     Y_Bt[i] = sp.Subs(Y_B, t, T[i])

#     phi_t[i] = sp.Subs(phi, t, T[i])



    


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


anim = FuncAnimation(fig, anima, frames=1000, interval=1, blit=False, repeat=True)


fig_graph = plt.figure(figsize=[13, 7])
ax_graph = fig_graph.add_subplot(2,3, 1)
ax_graph.plot(T, phi)
ax_graph.set_title("phi(t)")
ax_graph.set(xlim=[0, 10])
ax_graph.grid(True)

ax_graph = fig_graph.add_subplot(2, 3, 2)
ax_graph.plot(T, thetta)
ax_graph.set_title("thetta(t)")
ax_graph.set(xlim=[0, 10])
ax_graph.grid(True)

ax_graph = fig_graph.add_subplot(2, 3, 3)
ax_graph.plot(T, dphi)
ax_graph.set_title("phi'(t)")
ax_graph.set(xlim=[0, 10])
ax_graph.grid(True)

ax_graph = fig_graph.add_subplot(2, 3, 4)
ax_graph.plot(T, dthetta)
ax_graph.set_title("thetta'(t)")
ax_graph.set(xlim=[0, 10])
ax_graph.grid(True)

Rx = m * l * (ddthetta * np.cos(thetta) - dthetta**2*np.sin(thetta) - ddphi*np.cos(phi) + dphi**2*np.sin(phi))
Ry = m * l * (ddthetta * np.sin(thetta) + dthetta**2*np.cos(thetta) - ddphi*np.cos(phi) - dphi**2*np.cos(phi)) + m*g


ax_graph = fig_graph.add_subplot(2, 3, 5)
ax_graph.plot(T, Rx)
ax_graph.set_title("Rx(t)")
ax_graph.grid(True)

ax_graph = fig_graph.add_subplot(2, 3, 6)
ax_graph.plot(T, Ry)
ax_graph.set_title("Ry(t)")
ax_graph.grid(True)

# fig_opora = plt.figure(figsize=[13, 7])
# ax_opora = fig_graph.add_subplot(2, 2, 1)
# ax_opora.plot(T, ddphi)
# ax_opora.set_title("phi''(t)")
# ax_opora.set(xlim=[0, 10])
# ax_opora.grid(True)
# print(ddphi)

plt.show()