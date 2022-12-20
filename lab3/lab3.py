import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as pat
import sympy as sp
from scipy.integrate import odeint
import math



def formY(y, t, fomega_phi, fomega_thetta):
    y1, y2, y3, y4 = y
    dydt = [y3, y4, fomega_phi(y1, y2, y3, y4), fomega_thetta(y1, y2, y3, y4)]
    return dydt

#time
t_fin = 600

Steps = 10000

#constants
l = OA = AB = 5
R_Circle = l / 10
k1 = 5
k2 = 12
m = 10
c = 20
g = 9.81

#starting postion
phi0 = 2
thetta0 = 0
dphi0 = 1
dthetta0 = 1

t = sp.Symbol('t')

phi =  sp.Function('phi')(t) 
thetta = sp.Function('thetta')(t)
omega_phi = sp.Function('omega_phi')(t) 
omega_thetta = sp.Function('omega_thetta')(t)

#1 defining the kinetic energy
Vb2 = ((l * omega_phi) ** 2) + ((l * omega_thetta) ** 2) - 2 * l * l * omega_phi * omega_thetta * sp.cos(phi - thetta)
K = m * Vb2 / 2

#2 defining the potential energy
Pmg = m * g * l * (sp.cos(phi + 3.14/2) - sp.cos(thetta))
P2 = c * phi * phi / 2
P = Pmg + P2

#Not potential force
Q1 = - k1 * omega_phi 
Q2 = - k2 * omega_thetta

#Lagrange function
L = K - P

#equations
ur1 = sp.diff(sp.diff(L,omega_phi),t)-sp.diff(L,phi) - Q1
ur2 = sp.diff(sp.diff(L,omega_thetta),t)-sp.diff(L,thetta) - Q2

# isolating second derivatives(dV/dt and dom/dt) using Kramer's method
a11 = ur1.coeff(sp.diff(omega_phi, t), 1)
a12 = ur1.coeff(sp.diff(omega_thetta, t), 1)
a21 = ur2.coeff(sp.diff(omega_phi, t), 1)
a22 = ur2.coeff(sp.diff(omega_thetta, t), 1)
b1 = -(ur1.coeff(sp.diff(omega_phi, t), 0)).coeff(sp.diff(omega_thetta, t),
                                                     0).subs([(sp.diff(phi, t), omega_phi),
                                                      (sp.diff(thetta, t), omega_thetta)])
b2 = -(ur2.coeff(sp.diff(omega_phi, t), 0)).coeff(sp.diff(omega_thetta, t),
                                                     0).subs([(sp.diff(phi, t), omega_phi),
                                                     (sp.diff(thetta, t), omega_thetta)])
#we can check the result with:

# a11 = 1
# a12 = sp.cos(phi - thetta)
# a22 = 1
# a21 = sp.cos(phi - thetta)
# b1 = omega_thetta * omega_thetta*sp.sin(phi - thetta) + (g/l) * sp.sin(phi) - (c*phi + k1* omega_phi) / (m*l*l)
# b2 = -omega_phi*omega_phi*sp.sin(phi - thetta) - (g/l)*sp.sin(thetta) - k2*omega_thetta/(m*l*l)

print(b1)
print(b2)

detA = a11*a22-a12*a21
detA1 = b1*a22-b2*a12
detA2 = a11*b2-b1*a21

domega_phidt = detA1/detA
domega_thettadt = detA2/detA

Time = np.linspace(0, t_fin, Steps)

fomega_phi = sp.lambdify([phi, thetta, omega_phi, omega_thetta], domega_phidt, "numpy")
fomega_thetta = sp.lambdify([phi, thetta, omega_phi, omega_thetta], domega_thettadt, "numpy")

y0 = [phi0, thetta0, dphi0, dthetta0]

sol = odeint(formY, y0, Time, args=(fomega_phi, fomega_thetta))

Phi = sol[:, 0]
Thetta = sol[:, 1]
Omega_phi = sol[:, 2]
Omega_thetta = sol[:, 3]



X_A = sp.lambdify(phi, l * sp.cos(phi + 3.14/2))
Y_A = sp.lambdify(phi, l * sp.sin(phi + 3.14/2))

X_B = sp.lambdify([phi, thetta], l * sp.cos(phi + 3.14/2) + l * sp.sin(thetta))
Y_B = sp.lambdify([phi, thetta], l * sp.sin(phi + 3.14/2) - l * sp.cos(thetta))

XA = X_A(sol[:, 0])
YA = Y_A(sol[:, 0])

XB = X_B(sol[:, 0], sol[:, 1])
YB = Y_B(sol[:, 0], sol[:, 1])

Nv= 3
R1 = 0.001
R2 = l / 12


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.axis('equal')
ax.set(xlim=[-l * 2 - R_Circle - l/10, l *2 + R_Circle + l/10], ylim=[-l*2 - R_Circle - l/10,l*2 + R_Circle + l/10])

   
V_B = sp.lambdify([phi, thetta, omega_phi, omega_thetta], sp.sqrt(((l * omega_phi) ** 2) + ((l * omega_thetta) ** 2) - 2 * l * l * omega_phi * omega_thetta * sp.cos(phi - thetta)))
VB = V_B(Phi, Thetta, Omega_phi, Omega_thetta)
alpha = np.linspace(0, Nv*6.283+Phi[0]+ 3.14/2, 100)
X_SpiralSpr = -(R1 + alpha * (R2 - R1) / alpha[-1]) * np.sin(alpha)
Y_SpiralSpr = (R1 + alpha * (R2 - R1) / alpha[-1]) * np.cos(alpha)

beta = np.linspace(0, 2*math.pi, 100)

X_Circle = R_Circle * np.cos(beta)
Y_Circle = R_Circle * np.sin(beta)

fig_for_graphs = plt.figure(figsize=[13, 7])
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(Time, Phi, color='blue')
ax_for_graphs.set_title("Phi(t)")
ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(Time, Thetta, color='red')
ax_for_graphs.set_title('Va(t)')
ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2,2,3)
ax_for_graphs.plot(Time, Omega_phi * l, color='green')
ax_for_graphs.set_title("phi'(t) = omega_phi(t)")
ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 4)
ax_for_graphs.plot(Time, VB, color='black')
ax_for_graphs.set_title("Vb(t)")

ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

OX = ax.plot([-l * 2, l*2], [-l/10,-l/10], 'black', linestyle = '--')
Draw_Spring = ax.plot(X_SpiralSpr, Y_SpiralSpr, color='#666666')[0]
Draw_OA=ax.plot([0, XA[0]], [0, YA[0]], color='#808080')[0]
Draw_AB=ax.plot([XA[0], XB[0]], [YA[0], YB[0]], color='#808080' )[0]
PointB = ax.plot(XB[0], YB[0])[0]
PointA = ax.plot(XA[0], YA[0], color='#a0a0a0', marker='o')[0]
Draw_Circle = ax.plot(X_Circle + XB[0], Y_Circle + YB[0], color='black', linewidth=1)[0]
triangle = pat.Polygon([(0,0), (-l/10, -l/10), (l/10, -l/10)], color='#d3d3d3')
ax.add_patch(triangle)

def update(i):
    PointB.set_data(XB[i],YB[i])
    Draw_OA.set_data([0, XA[i]], [0, YA[i]])
    PointA.set_data(XA[i], YA[i])
    Draw_AB.set_data([XA[i], XB[i]], [YA[i], YB[i]])
    Draw_Circle.set_data(X_Circle + XB[i], Y_Circle + YB[i])
    alpha = np.linspace(0, Nv*6.28+Phi[i] + 3.14/2, 100)
    X_SpiralSpr = -(R1 + alpha * (R2 - R1) / alpha[-1]) * np.sin(alpha - 1.57)
    Y_SpiralSpr = (R1 + alpha * (R2 - R1) / alpha[-1]) * np.cos(alpha - 1.57)
    Draw_Spring.set_data(X_SpiralSpr, Y_SpiralSpr)
    return [PointB, Draw_OA, Draw_Spring, Draw_AB, PointA, Draw_Circle]

anima = FuncAnimation(fig, update, frames=Steps, interval=1)
plt.show()