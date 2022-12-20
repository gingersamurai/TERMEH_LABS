import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math


def chetv(x, y):  # функция, определяющая в какой четверти координат находится вектор
    if x >= 0 and y >= 0:
        return 1
    if x <= 0 and y >= 0:
        return 2
    if x <= 0 and y <= 0:
        return 3
    if x >= 0 and y <= 0:
        return 4


def compare_tan(x0, y0, x1, y1, ch):  # функция, сравнивающая тангенсы векторов, находящихся в одной четверти
    if x1 == 0:
        t1 = 999999
    else:
        t1 = y1 / x1
    if x0 == 0:
        t0 = 999999
    else:
        t0 = y0 / x0
    if ch % 2:
        if t1 < t0:
            return 1
        else:
            return -1
    else:
        if t1 < t0:
            return -1
        else:
            return 1


def direction(x0, y0, x1, y1):  # функция, определяющая в каком направлении должен быть направлен радиус кривизны
    '''
    Функция сравнивает текущий вектор с предыдущим. Если вектора находятся в одной четверти или находятся в четвертях
    1/3 или 2/4, то сравниваются их тангенсы. Для векторов, находящихся в четвертях 1-2, 2-3, 3-4, 4-1, выводится -1.
    Для векторов 2-1, 3-2, 4-3, 1-4 выводится 1.
    1 - движение по часовой стрелке, радиус кривизны должен быть слева
    -1 - движение против часовой стрелки, радиус кривизны должен быть справа
    '''
    ch0 = chetv(x0, y0)
    ch1 = chetv(x1, y1)
    if ch1 == ch0:
        return compare_tan(x0, y0, x1, y1, ch1)
    else:
        if (ch0 == 1 and ch1 == 2) or (ch0 == 2 and ch1 == 3) or (ch0 == 3 and ch1 == 4) or (ch0 == 4 and ch1 == 1):
            return -1
        elif (ch1 % 2) == (ch0 % 2):
            return -compare_tan(x0, y0, x1, y1, ch1)
        else:
            return 1


def Rotation(X, Y, alpha):  # функция для поворота стрелок
    RX = X * np.cos(alpha) - Y * np.sin(alpha)
    RY = X * np.sin(alpha) + Y * np.cos(alpha)
    return RX, RY


def animation(i):  # функция анимации
    P.set_data(X[i], Y[i])
    VLine.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    ALine.set_data([X[i], X[i] + AX[i]], [Y[i], Y[i] + AY[i]])
    d = direction(X[i - 1], Y[i - 1], X[i], Y[i])
    CVector.set_data([X[i], X[i] + NX[i] * CV[i] * d], [Y[i], Y[i] + NY[i] * CV[i] * d])
    RVector.set_data([0, X[i]], [0, Y[i]])

    RArrowVX, RArrowVY = Rotation(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowVX + X[i] + VX[i], RArrowVY + Y[i] + VY[i])
    RArrowAX, RArrowAY = Rotation(ArrowX, ArrowY, math.atan2(AY[i], AX[i]))
    AArrow.set_data(RArrowAX + X[i] + AX[i], RArrowAY + Y[i] + AY[i])
    RArrowCX, RArrowCY = Rotation(ArrowX, ArrowY, math.atan2(d * NY[i], d * NX[i]))
    CVArrow.set_data(RArrowCX + X[i] + NX[i] * CV[i] * d, RArrowCY + Y[i] + NY[i] * CV[i] * d)
    RArrowRVX, RArrowRVY = Rotation(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    RVArrow.set_data(RArrowRVX + X[i], RArrowRVY + Y[i])
    return P, VLine, ALine, VArrow, AArrow, CVector, CVArrow, RVector, RVArrow,


t = sp.Symbol('t')
R = 75
S = 4

r = 2 + sp.sin(6 * t)
fi = 6.5 * t + 1.2 * sp.cos(6 * t)

x = r * sp.cos(fi)
y = r * sp.sin(fi)

Vx = sp.diff(x, t)  # проекция скорости на х
Vy = sp.diff(y, t)  # проекция скорости на у
Ax = sp.diff(Vx, t)  # проекция ускорения на х
Ay = sp.diff(Vy, t)  # проекция ускорения на у
V = sp.sqrt(Vx ** 2 + Vy ** 2)  # модуль скорости
Afull = sp.sqrt(Ax ** 2 + Ay ** 2)  # полное ускорение
Atan = sp.diff(V)  # тангенциальное ускорение
An = sp.sqrt(Afull ** 2 - Atan ** 2)  # нормальное ускорение
Сurve = V ** 2 / An  # длина радиуса кривизны

T = np.linspace(0, 8, 1500)  # нтервал времени 8 секунд, разделенных на 1500 частей

X = np.zeros_like(T)  # нулевые массивы размером 1500 для функций
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)
NX = np.zeros_like(T)
NY = np.zeros_like(T)
CV = np.zeros_like(T)

for i in np.arange(len(T)):  # вычисление значений функций в i момент времени
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    AX[i] = sp.Subs(Ax, t, T[i])
    AY[i] = sp.Subs(Ay, t, T[i])
    NX[i] = VY[i] / math.sqrt(VX[i] ** 2 + VY[i] ** 2)  # проекция единичного вектора, перпендикулярного V на х
    NY[i] = -VX[i] / math.sqrt(VX[i] ** 2 + VY[i] ** 2)  # проекция единичного вектора, перпендикулярного V на у
    CV[i] = sp.Subs(Сurve, t, T[i])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-5, 15], ylim=[-R, R])

ax1.plot(X, Y)

P, = ax1.plot(X[0], Y[0], marker='o')  # точка
VLine, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')  # вектор скорости
ALine, = ax1.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], 'y')  # вектор ускорения
CVector, = ax1.plot([X[0], X[0] + NX[0] * CV[0]], [Y[0], Y[0] + NY[0] * CV[0]], 'orange')  # вектор кривизны
RVector, = ax1.plot([0, X[0]], [0, Y[0]], 'black')  # радиус-вектор

ArrowX = np.array([-0.2 * S, 0, -0.2 * S])
ArrowY = np.array([0.1 * S, 0, -0.1 * S])
RArrowVX, RArrowVY = Rotation(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowVX + X[0] + VX[0], RArrowVY + Y[0] + VY[0], 'r')  # стрелка для вектора скорости
RArrowAX, RArrowAY = Rotation(ArrowX, ArrowY, math.atan2(AY[0], AX[0]))
AArrow, = ax1.plot(RArrowAX + X[0] + AX[0], RArrowAY + Y[0] + AY[0], 'y')  # стрелка для вектора ускорения
RArrowCX, RArrowCY = Rotation(ArrowX, ArrowY, math.atan2(NY[0], NX[0]))
CVArrow, = ax1.plot(RArrowCX + X[0] + NX[0] * CV[0],
                    RArrowCY + Y[0] + NY[0] * CV[0], 'orange')  # стрелка для вектора кривизны
RArrowRVX, RArrowRVY = Rotation(ArrowX, ArrowY, math.atan2(Y[0], X[0]))
RVArrow, = ax1.plot(RArrowRVX + X[0], RArrowRVY + Y[0], 'black')  # стрелка для радиус-вектора

anim = FuncAnimation(fig, animation, frames=1500, interval=6, blit=True, repeat=False)

plt.show()