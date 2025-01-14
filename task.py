import sys
import numpy as np
from numba import njit, prange
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import HDFStorage
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)

Nt = np.int64(100000)
t = np.float64(0)
dt = np.float64(.001)
Cr = np.float64(.5)
counter = 0
save_par = 16

rlb, rrb = np.float64(0.), np.float64(3.) 
zlb, zrb = np.float64(0.), np.float64(7.) 
dr, dz = np.float64(.005), np.float64(.005) 
drdz = dr * dz
Crh = Cr * np.min([dr, dz])

g, phi, chi = np.true_divide(11.,10.), np.float64(.009), np.float64(10.)

#чтобы лишний раз не вычислять
dchi = np.true_divide(1., chi)
phidz = np.multiply(phi, dz)
chiphidz = np.multiply(chi, phidz)
g1, g3 = np.subtract(g, 1.), np.subtract(g, 3.)
gg1 = g * g1
mg1, mg3 = np.negative(g1), np.negative(g3)
g1d5, g3d5 = np.multiply(g1, .5), np.multiply(g3, .5)
dg1 = np.true_divide(1., g1)
gphi = np.multiply(g, phi)
odr, odz = np.true_divide(1., dr), np.true_divide(1., dz)
hodr, hodz = np.true_divide(.5, dr), np.true_divide(.5, dz)

#разбиение области
r, z = np.arange(rlb, rrb + dr, dr), np.arange(zlb, zrb + dz, dz)
Nr, Nz = np.int64(len(r)), np.int64(len(z))
Nrm1, Nzm1 = Nr - 1, Nz - 1
Nrm2, Nzm2 = Nr - 2, Nz - 2

#массив координат узлов; [i,j,0] - r i,j-ого узла, [i,j,1] - z -//-
rz = np.stack((np.meshgrid(z,r)[1],np.meshgrid(z,r)[0]), axis = 2)
rd, zd = np.meshgrid(r, z)

#массивы физических переменных
u = np.zeros([Nr,Nz], dtype = np.float64)
v, rho, eint = np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)

#консервативные переменные и их функции
Q = np.zeros([4,Nr,Nz], dtype = np.float64)
Qtmp = np.zeros_like(Q)
F, G, D = np.zeros_like(Q), np.zeros_like(Q), np.zeros_like(Q)

iDeltas, jDeltas = np.zeros_like(Q), np.zeros_like(Q)
iMinmods, jMinmods = np.zeros_like(Q), np.zeros_like(Q)

QiR, QiL = np.zeros_like(Q), np.zeros_like(Q)# потоки слева и справа
QjR, QjL = np.zeros_like(Q), np.zeros_like(Q)# сверху и снизу
fQiR, fQiL = np.zeros_like(Q), np.zeros_like(Q)# f(QiR), f(QiL)
gQjR, gQjL = np.zeros_like(Q), np.zeros_like(Q)# g(QjR), g(QjL)
Qih, Qjh = np.zeros_like(Q), np.zeros_like(Q)# интерполяции Q на полуцелые индексы
Ft, Gt = np.zeros_like(Q), np.zeros_like(Q) # f, g с тильдами
SRf, SRg = np.zeros_like(u), np.zeros_like(u)# спектральные радиусы df/dq в i+-1/2, j и dg/dq в i, j+-1/2

#начальные условия
Ad = np.float64(20.)
Bd = np.true_divide(-.3, np.sqrt(np.true_divide(2.,3.)))*np.sqrt(g1)
zc = np.float64(3.)
d = np.float64(.3)
d2 = d * d
zd = np.float64(3.5)
jzc = np.where(np.isclose(z,zc))[0][0]
jzd = np.where(np.isclose(z,zd))[0][0]


for j in range(jzc+1):
    rho[:,j] = chi * np.power(1. - chiphidz * (j - jzc), dg1)
    eint[:,j] = dchi * np.power(dchi * rho[:,j], g1)
for j in range(jzc+1,Nz):
    for i in range(Nr):
        rho[i,j] = np.power(1. - phidz * (j - jzc), dg1) + Ad * np.exp(-(np.power(dz * (j - jzd), 2.) + np.power(dr * i, 2.)) / d2)
        eint[i,j] = np.power(rho[i,j], -1.)
        v[i,j] = Bd * np.exp(-(np.power(dz * (j - jzd), 2.) + np.power(dr * i, 2.))/ d2)


@njit(parallel = True)
def primitive_to_conservative_nb(Q, rho, u, v, eint, rz):
    for i in prange(Nr):
        for j in prange(Nz):
            Q[0,i,j] = rho[i,j] * rz[i,j,0]
            Q[1,i,j] = u[i,j] * Q[0,i,j]
            Q[2,i,j] = v[i,j] * Q[0,i,j]
            Q[3,i,j] = (eint[i,j] + .5 * (u[i,j] * u[i,j] + v[i,j] * v[i,j])) * Q[0,i,j]
    return Q


@njit(parallel = True)
def conservative_to_primitive_nb(rho, u, v, eint, Q, rz):
    for i in prange(1,Nrm1):
        for j in prange(1,Nzm1):
            rho[i,j] = Q[0,i,j] / rz[i,j,0]
            u[i,j] = Q[1,i,j] / Q[0,i,j]
            v[i,j] = Q[2,i,j] / Q[0,i,j]
            eint[i,j] = Q[3,i,j] / Q[0,i,j] - .5 * (u[i,j] * u[i,j] + v[i,j] * v[i,j])
    return rho, u, v, eint


@njit(parallel = True)
def calculate_FGD_full_nb(F, G, D, Q, rho, u, v, eint, rz):
    for i in prange(1,Nr):
        for j in prange(Nz):
            Q21 = Q[1,i,j] / Q[0,i,j]
            Q31 = Q[2,i,j] / Q[0,i,j]
            Q41 = Q[3,i,j] / Q[0,i,j]
            Q212 = Q21 * Q21
            Q312 = Q31 * Q31
            M4 = g * Q41 - g1d5 * (Q212 + Q312)
            F[0,i,j] = Q[1,i,j]
            F[1,i,j] = Q[0,i,j] * (-g3 * .5 * Q212 + g1 * (Q41 - .5 * Q312)) 
            F[2,i,j] = Q[1,i,j] * Q31
            F[3,i,j] = Q[1,i,j] * M4
            G[0,i,j] = Q[2,i,j]
            G[1,i,j] = F[2,i,j]
            G[2,i,j] = Q[0,i,j] * (-g3 * .5 * Q312 + g1 * (Q41 - .5 * Q212)) 
            G[3,i,j] = Q[2,i,j] * M4
            D[0,i,j] = 0.
            D[1,i,j] = eint[i,j] * rho[i,j] * g1
            D[2,i,j] = -rho[i,j] * rz[i,j,0] * gphi
            D[3,i,j] = -rho[i,j] * v[i,j] * rz[i,j,0] * gphi
    return F, G, D


@njit(parallel = True)
def predictor_nb(Qtmp, Q, F, G, D, dt):
    hdt = .5 * dt
    for i in prange(1,Nrm1):
        for j in prange(1, Nzm1):
            Qtmp[:,i,j] = Q[:,i,j] - hdt * (hodr * (F[:,i+1,j] - F[:,i-1,j]) + hodz * (G[:,i,j+1] - G[:,i,j-1]) - D[:,i,j])
    return Qtmp


@njit(parallel = True)
def satisfy_boundary_conditions_predictor_nb(rho, u, v, eint):
    for i in prange(1,Nrm1): #на верхней и нижней границе (z = 0, z = 7)
        rho[i,0], rho[i,-1] = rho[i,1], rho[i,-2] #мягкие
        u[i,0], u[i,-1] = u[i,1], u[i,-2]
        v[i,0], v[i,-1] = 0., 0.
        eint[i,0], eint[i,-1] = eint[i,1], eint[i,-2]
    for j in prange(Nz): #слева и справа
        rho[0,j], rho[-1,j] = rho[1,j], rho[-2,j]
        u[0,j], u[-1,j] = 0., 0.
        v[0,j], v[-1,j] = 0., v[-2,j]
        eint[0,j], eint[-1,j] = eint[1,j], eint[-2,j]
    return rho, u, v, eint


@njit(parallel = True)
def calculate_deltas_full_nb(iDeltas, jDeltas, Qtmp):
    for i in prange(Nrm1):
        for j in prange(Nzm1):
            iDeltas[:,i,j] = Qtmp[:,i+1,j] - Qtmp[:,i,j]
            jDeltas[:,i,j] = Qtmp[:,i,j+1] - Qtmp[:,i,j]
    return iDeltas, jDeltas


@njit(parallel = True)
def calculate_minmods_full_nb(iMinmods, jMinmods, iDeltas, jDeltas):
    for i in prange(1,Nrm1):
        for j in prange(1,Nzm1):
            for k in prange(4):
                signx = np.sign(iDeltas[k,i-1,j])
                modx = np.abs(iDeltas[k,i-1,j])
                ysignx = iDeltas[k,i,j] * np.sign(iDeltas[k,i-1,j])
                minb = np.min(np.array([modx, ysignx]))
                maxb = np.max(np.array([0., minb]))
                iMinmods[k,i,j] = signx * maxb

                signx = np.sign(jDeltas[k,i,j-1])
                modx = np.abs(jDeltas[k,i,j-1])
                ysignx = jDeltas[k,i,j] * np.sign(jDeltas[k,i,j-1])
                minb = np.min(np.array([modx, ysignx]))
                maxb = np.max(np.array([0., minb]))
                jMinmods[k,i,j] = signx * maxb
    return iMinmods, jMinmods


@njit(parallel = True)
def calculate_qrl_full_nb(QiR, QiL, QjR, QjL, Qtmp, iMinmods, jMinmods):
    for i in prange(1,Nrm1):
        for j in prange(1,Nzm1):
            QiR[:,i,j] = Qtmp[:,i,j] - .5 * iMinmods[:,i,j]
            QiL[:,i,j] = Qtmp[:,i,j] + .5 * iMinmods[:,i,j]
            QjR[:,i,j] = Qtmp[:,i,j] - .5 * jMinmods[:,i,j]
            QjL[:,i,j] = Qtmp[:,i,j] + .5 * jMinmods[:,i,j]
    return QiR, QiL, QjR, QjL


@njit(parallel = True)
def calculate_f_full_nb(F, Q):
    for i in prange(1,Nrm1):
        for j in prange(1,Nzm1):
            dQ1 = 1. / Q[0,i,j]
            Q21 = Q[1,i,j] * dQ1
            Q31 = Q[2,i,j] * dQ1
            Q41 = Q[3,i,j] * dQ1
            Q212 = Q21 * Q21
            Q312 = Q31 * Q31
            M4 = g * Q41 - g1d5 * (Q212 + Q312)
            F[0,i,j] = Q[1,i,j]
            F[1,i,j] = Q[0,i,j] * (-g3d5 * Q212 + g1 * (Q41 - .5 * Q312)) 
            F[2,i,j] = Q[1,i,j] * Q31
            F[3,i,j] = Q[1,i,j] * M4
    return F


@njit(parallel = True)
def calculate_g_full_nb(G, Q):
    for i in prange(1,Nrm1):
        for j in prange(1,Nzm1):
            dQ1 = 1. / Q[0,i,j]
            Q21 = Q[1,i,j] * dQ1
            Q31 = Q[2,i,j] * dQ1
            Q41 = Q[3,i,j] * dQ1
            Q212 = Q21 * Q21
            Q312 = Q31 * Q31
            M4 = g * Q41 - g1d5 * (Q212 + Q312)
            G[0,i,j] = Q[2,i,j]
            G[1,i,j] = Q[1,i,j] * Q31
            G[2,i,j] = Q[0,i,j] * (-g3d5 * Q312 + g1 * (Q41 - .5 * Q212)) 
            G[3,i,j] = Q[2,i,j] * M4
    return G


#считаем интерполяцию Q в полуцелых узлах, мб надо параболическую
@njit(parallel = True)
def linear_interpolation(Q, Qih, Qjh):
    for i in prange(Nrm1):
        Qih[:,i,:] = .5 * (Q[:,i,:] + Q[:,i+1,:])
    for j in prange(Nzm1):
        Qjh[:,:,j] = .5 * (Q[:,:,j] + Q[:,:,j+1])
    return Qih, Qjh


@njit(parallel = True)
def calculate_SRs_full_nb(SRf, SRg, Qih, Qjh):#проверить конечно же
    for i in prange(Nrm1):
        for j in prange(Nzm1):
            dQ1 = 1. / Qih[0,i,j]
            Q21 = Qih[1,i,j] * dQ1 
            Q31 = Qih[2,i,j] * dQ1
            Q41 = Qih[3,i,j] * dQ1
            c = np.sqrt(gg1 * (Q41 - .5 * (Q21 * Q21 + Q31 * Q31)))
            SRf[i,j] = np.max(np.absolute(np.array([Q21, Q21 + c, Q21 - c])))
                        
            dQ1 = 1. / Qjh[0,i,j]
            Q21 = Qjh[1,i,j] * dQ1 
            Q31 = Qjh[2,i,j] * dQ1
            Q41 = Qjh[3,i,j] * dQ1
            c = np.sqrt(gg1 * (Q41 - .5 * (Q21 * Q21 + Q31 * Q31)))
            SRg[i,j] = np.max(np.absolute(np.array([Q31, Q31 + c, Q31 - c])))
    return SRf, SRg


@njit(parallel = True)
def calculate_tildas_full_nb(Ft, Gt, fQiR, fQiL, gQjR, gQjL, SRf, SRg, QiR, QiL, QjR, QjL):
    for i in prange(1,Nrm2):
        for j in prange(1,Nzm2):
            Ft[:,i,j] = .5 * (fQiR[:,i+1,j] + fQiL[:,i,j] - 
                            SRf[i,j] * (QiR[:,i+1,j] - QiL[:,i,j])) 
            Gt[:,i,j] = .5 * (gQjR[:,i,j+1] + gQjL[:,i,j] - 
                            SRg[i,j] * (QjR[:,i,j+1] - QjL[:,i,j])) 
    return Ft, Gt


@njit(parallel = True)
def corrector_nb(Q, Ft, Gt, D, dt):
    for i in prange(2,Nrm2):
        for j in prange(2,Nzm2):
            Q[:,i,j] = Q[:,i,j] - dt * ( odr * (
                Ft[:,i,j] - Ft[:,i-1,j]) + odz * (
                    Gt[:,i,j] - Gt[:,i,j-1]) - D[:,i,j])
    return Q


@njit(parallel = True)
def satisfy_boundary_conditions_corrector_nb(rho, u, v, eint):
    for i in prange(1,Nrm1):
        rho[i,1], rho[i,-2] = rho[i,2], rho[i,-3]
        u[i,1], u[i,-2] = u[i,2], u[i,-3]
        v[i,1], v[i,-2] = .5 * v[i,2], .5 * v[i,-3]#0., 0.
        eint[i,1], eint[i,-2] = eint[i,2], eint[i,-3] 
    for j in prange(1,Nzm1):
        rho[1,j], rho[-2,j] = rho[2,j], rho[-3,j]
        u[1,j], u[-2,j] = .5 * u[2,j], .5 * u[-3,j]#0., 0.
        v[1,j], v[-2,j] = v[2,j], v[-3,j]#0., v[-3,j]
        eint[1,j], eint[-2,j] = eint[2,j], eint[-3,j]
    for i in prange(Nr):
        rho[i,0], rho[i,-1] = rho[i,1], rho[i,-2]
        u[i,0], u[i,-1] = u[i,1], u[i,-2]
        v[i,0], v[i,-1] = 0., 0.
        eint[i,0], eint[i,-1] = eint[i,1], eint[i,-2] 
    for j in prange(Nz):
        rho[0,j], rho[-1,j] = rho[1,j], rho[-2,j]
        u[0,j], u[-1,j] = 0., 0.
        v[0,j], v[-1,j] = v[1,j], v[-2,j]#0., v[-2,j]
        eint[0,j], eint[-1,j] = eint[1,j], eint[-2,j]
    return rho, u, v, eint


def Courant_Friedrichs_Lewy_condition(dt, Q, SRf, SRg, Cr):
    SRf, SRg = calculate_SRs_full_nb(SRf, SRg, Q, Q)
    return Crh / np.max([np.max(SRf), np.max(SRg)])


def picture_test(arr):
    fig, ax = plt.subplots()
    im = ax.imshow(np.flipud(arr.T), cmap = cm.Greys,
                extent = [0.,3.,0.,7.],
                vmax = arr.max(), vmin = arr.min())
    ax.set(xlabel = 'r', ylabel = 'z')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.suptitle('test')
    plt.colorbar(im, cax=cax)
    plt.show()
    plt.close()
    return

Z, R = np.meshgrid(z, r)

hdf = HDFStorage('data-1.hdf5', 'f8', 16)
hdf.append(t = t, u = u, v = v, rho = rho, eint = eint)
hdf.write(r = R, z = Z, dr = dr, dz = dz)
    
Q = primitive_to_conservative_nb(Q, rho, u, v, eint, rz)

if __name__ == '__main__':
    Q = primitive_to_conservative_nb(Q, rho, u, v, eint, rz)
    for _ in tqdm(range(1,Nt+1), ascii = True, desc = 'progress'):
        hdf.watch(t = t, u = u, v = v, rho = rho, eint = eint)

        dt = Courant_Friedrichs_Lewy_condition(dt, Q, SRf, SRg, Cr)

        F, G, D = calculate_FGD_full_nb(F, G, D, Q, rho, u, v, eint, rz)

        Qtmp = predictor_nb(Qtmp, Q, F, G, D, dt)
        rho, u, v, eint = conservative_to_primitive_nb(rho, u, v, eint, Qtmp, rz)
        rho, u, v, eint = satisfy_boundary_conditions_corrector_nb(rho, u, v, eint)
        Qtmp = primitive_to_conservative_nb(Qtmp, rho, u, v, eint, rz) 
 
        iDeltas, jDeltas = calculate_deltas_full_nb(iDeltas, jDeltas, Qtmp)
        iMinmods, jMinmods = calculate_minmods_full_nb(iMinmods, jMinmods, iDeltas, jDeltas)

        QiR, QiL, QjR, QjL = calculate_qrl_full_nb(QiR, QiL, QjR, QjL, Qtmp, iMinmods, jMinmods)
        fQiR = calculate_f_full_nb(fQiR, QiR)
        fQiL = calculate_f_full_nb(fQiL, QiL)
        gQjR = calculate_g_full_nb(gQjR, QjR)
        gQjL = calculate_g_full_nb(gQjL, QjL)

        Qih, Qjh = linear_interpolation(Q, Qih, Qjh)#linear_interpolation(Qtmp, Qih, Qjh)
        SRf, SRg = calculate_SRs_full_nb(SRf, SRg, Qih, Qjh)
        Ft, Gt = calculate_tildas_full_nb(Ft, Gt, fQiR, fQiL, gQjR, gQjL, SRf, SRg, QiR, QiL, QjR, QjL)
        Q = corrector_nb(Q, Ft, Gt, D, dt)
        rho, u, v, eint = conservative_to_primitive_nb(rho, u, v, eint, Q, rz)
        rho, u, v, eint = satisfy_boundary_conditions_corrector_nb(rho, u, v, eint)
        Q = primitive_to_conservative_nb(Q, rho, u, v, eint, rz)
        t += dt
        counter += 1
        
exit()