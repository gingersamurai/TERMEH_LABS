import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    print(dy)
    return dy

t = sp.symbols('t')

m = 0.5 	
l = 5		
c = 8
k1 = 2
k2 = 5
g = 9.81
phi0 = np.pi / 2
thetta0 = 0
dphi0 = 0
dthetta0 = 0




y0 = [phi0, thetta0, dphi0, dthetta0]

T = np.linspace(0, 10, 1000)


Y = odeint(odesys, y0, T, (g, l, c, k1, k2, m))

phi = Y[:, 0]
thetta = Y[:, 1]
dphi = Y[:, 2]
dthetta = Y[:, 3]
# phi = t
# thetta = 2 * np.pi * sp.sin(t)
# print(phi)

OA = l
AB = l

X_A = -1 * OA * np.sin(phi)
Y_A = 1 * OA * np.cos(phi)

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

X_sprt, Y_sprt = Rod2D(X_sprt, Y_sprt, np.pi/2)

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
pt_A, = ax_main.plot(X_At[0], Y_At[0], marker='o', color='red') #A
pt_B, = ax_main.plot(X_Bt[0], Y_Bt[0], marker='o', markersize=10, color='magenta') #B

pt_spr, = ax_main.plot(X_sprt, Y_sprt)

print(phi[:100])

def anima(i):
    pt_A.set_data(X_At[i], Y_At[i])
    pt_B.set_data(X_Bt[i], Y_Bt[i])

    pt_OA.set_data([0, X_At[i]], [0, Y_At[i]])
    pt_AB.set_data([X_At[i], X_Bt[i]], [Y_At[i], Y_Bt[i]])

    new_spr_x, new_spr_y = Rod2D(X_sprt, Y_sprt, phi_t[i])
    pt_spr.set_data(new_spr_x, new_spr_y)

    return pt_A, pt_B, pt_OA, pt_AB, pt_spr


anim = FuncAnimation(fig, anima, frames=1000, interval=10, blit=True, repeat=True)


fig_graph = plt.figure(figsize=[13, 7])
ax_graph = fig_graph.add_subplot(2, 2, 1)
ax_graph.plot(T, phi)
ax_graph.set_title("phi(t)")
ax_graph.set(xlim=[0, T[-1]])
ax_graph.grid(True)

ax_graph = fig_graph.add_subplot(2, 2, 2)
ax_graph.plot(T, thetta)
ax_graph.set_title("thetta(t)")
ax_graph.set(xlim=[0, T[-1]])
ax_graph.grid(True)

ax_graph = fig_graph.add_subplot(2, 2, 3)
ax_graph.plot(T, dphi)
ax_graph.set_title("phi'(t)")
ax_graph.set(xlim=[0, T[-1]])
ax_graph.grid(True)

ax_graph = fig_graph.add_subplot(2, 2, 4)
ax_graph.plot(T, dthetta)
ax_graph.set_title("thetta'(t)")
ax_graph.set(xlim=[0, T[-1]])
ax_graph.grid(True)



plt.show()