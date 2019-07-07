import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from matplotlib import colors
import pandas as pd 
from numpy.random import rand
'''This program has a multitude of methods to numerically solve the 3-dimmensional ising
model probelm. Specifically there are methods for manipulating a spin-field, other methods 
for calculating values from this spin-field for different temperatures and spin-field sizes, 
and methods to plot these values obtained for different conditions of the spin-field.'''

#nearest neighbour interaction strength
J = 1

#creates a square spin-field with side length = nSites
def init(nSites):
    return np.random.choice([-1,1], size=(nSites, nSites, nSites))

#does (H^M)^N monte-carlo updates to the spin-field 
def ising_update(field, T):
    N, M, H = field.shape
    for _ in range(N):
        for _ in range(M):
            for _ in range(H):
                x = np.random.randint(N)
                y = np.random.randint(M)
                z = np.random.randint(H)
                spin = field[x,y,z]
                total = field[(x+1)%N,y,z] + field[(x-1)%N,y,z] + field[x,(y+1)%M,z] + \
                    field[x,(y-1)%M,z] + field[x,y,(z+1)%H] + field[x,y,(z-1)%H]   
                dE = 2*J*spin*total
                if dE < 0:
                    spin *= -1
                elif rand() < np.exp(-dE/T):
                    spin *= -1
                field[x,y,z] = spin
    return field

#calculates the energy of the spin-field
def energy(field):
    N, M, H = field.shape
    energy = 0
    for i in range(N):
        for j in range(M):
            for k in range(H):
                s = field[i,j,k]
                nb = field[(i+1)%N,j,k] + field[i,(j+1)%M,k] + field[i,j,(k+1)%H] +\
                    field[(i-1)%N,j,k] + field[i,(j-1)%M,k] + field[i,j,(k-1)%H]
                energy += -J*nb*s
    return energy/6.

#calculates the magnetization of the spin-field
def mag(field):
    mag = np.sum(field)
    return abs(mag)

#exports ['T','E','M','C','X','LOGM','U'] values to csv files named 
#[5_3D.csv, 10_3D.csv, 25_3D.csv] indicating the side length of the lattice
#and that it is in 3D, so a cube
def data():
    nSites = [5,10,25]
    for k in range(len(nSites)):
        start = 2.0
        stop = 5.5
        step = .1
        nt = int((stop-start)/step)
        N = nSites[k] 
        nEquil = 1240
        nSteps = 1240
        T = np.linspace(start, stop, nt)
        E, M, C, X, LOGM, U = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
        n1, n2 = 1.0/(nSteps*N*N*N), 1.0/(nSteps*nSteps*N*N*N)
        vol = N*N*N
        for tt in range(nt):
            E1 = M1 = E2 = M2 = M4 = 0
            iT = 1.0/(T[tt]); iT2 = iT*iT
            E1: np.float64
            E2: np.float64
            M1: np.float64
            M2: np.float64
            field = init(N)

            for _ in range(nEquil):
                ising_update(field, T[tt])

            for _ in range(nSteps):
                ising_update(field, T[tt])
                Ene = energy(field)
                Mag = mag(field)

                E1 = E1 + Ene
                M1 = M1 + Mag
                M2 = M2 + Mag*Mag*n1
                E2 = E2 + Ene*Ene*n1
                M4 = M4 + Mag*Mag*Mag*Mag*n1

            E[tt] = n1*E1
            M[tt] = n1*M1
            C[tt] = (E2 - n2*E1*E1)*iT2
            X[tt] = (M2 - n2*M1*M1)*iT
            U[tt] = 1 - (M4/3*(M2*M2)*vol)
            LOGM[tt] = np.log10(n1*M1) 
        data = np.column_stack((T,E,M,C,X,LOGM,U))
        df = pd.DataFrame(data, columns=['T','E','M','C','X','LOGM','U'])
        if k == 0:
            df.to_csv('5_3D.csv', index=False)
        elif k == 1:
            df.to_csv('10_3D.csv', index=False)
        else:
            df.to_csv('25_3D.csv', index=False)
    print("All Done!")

#plots m/n vs. T from "data" for different values of N
def mag_3D(data):
    color = ['red', 'blue', 'green']
    marker = ['s', '*', 'o']
    for i, d in enumerate(data):
        df = pd.read_csv(d)
        T = df['T']
        M = df['M']
        plt.scatter(T, M, c=color[i], marker=marker[i])
    plt.legend(['N = 5', 'N = 10', 'N = 25'], loc=3)
    plt.xlabel('T')
    plt.ylabel('$|<M>|/N$')
    plt.grid(b=True)
    plt.show()

#plots energy vs. T from "data" for different values of N
def energy_3D(data):
    color = ['red', 'blue', 'green']
    marker = ['s', '*', 'o']
    for i, d in enumerate(data):
        df = pd.read_csv(d)
        T = df['T']
        E = df['E']
        plt.scatter(T, E, color=color[i], marker=marker[i])
    plt.legend(['N = 5', 'N = 10', 'N = 25'], loc=2)
    plt.xlabel('T')
    plt.ylabel('Energy')
    plt.grid(b=True)
    plt.show()

#plots chi vs. T from "data" for different values of N
def chi_3D(data):
    color = ['red', 'blue', 'green']
    marker = ['s', '*', 'o']
    c = []
    t = []
    line = []
    for i, d in enumerate(data):
        df = pd.read_csv(d)
        T = df['T']
        chi = df['X']
        TC = 4.3
        width = 0.5
        dfn = df[(df['T'] < TC + width) & (df['T'] > TC - width)]
        maxx = max(dfn['X'])
        maxt = dfn[dfn['X'] == maxx]['T']
        c.append(maxx)
        t.append(maxt)
        l = plt.scatter(T, chi, color=color[i], marker=marker[i])
        line.append(l)
    plt.legend([line[0], line[1], line[2]], ['N = 5 with $C_{max}$ = %.2f at $T_{max}$ = %.2f' %(c[0], t[0]), 'N = 10 with $C_{max}$ = %.2f at $T_{max}$ = %.2f' %(c[1], t[1]), 'N = 25 with $C_{max}$ = %.2f at $T_{max}$ = %.2f' %(c[2], t[2])], loc=2)
    plt.xlabel('T')
    plt.ylabel('Magnetic Susceptibility')
    plt.xlim(3.5, 5.5)
    plt.axvline(x=TC-width)
    plt.axvline(x=TC+width)
    plt.grid(b=True)
    plt.show()

#rejects data points outside of m standard deviations from the mean
#Of course this should only be used on data that has desirerable points near the mean
def reject_outliers(data, m=0.5):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

#plot log(m/n) vs. T from "data" for different values of N
def logmag_3D(data):
    color = ['red', 'blue', 'green', 'orange', 'purple']
    f, ax = plt.subplots(1, 3, sharey=True)
    T = np.linspace(4.0, 5.0, 5)
    for i, d in enumerate(data):
        for j, tt in enumerate(T):
            df = pd.read_csv(d)
            df = df[df['T'] < tt]
            logt = np.log10(tt - df['T'])
            logm = df['LOGM']
            ax[i].scatter(logt, logm, color=color[j])
            if j == 1:
                line = np.polyfit(logt, logm, 1)
                line_f = np.poly1d(line)
                ax[i].plot(logt, line_f(logt), color=color[j])
                leg = ax[i].legend(['T = %.2f with slope %.2f' %(tt, line[0])], loc=3)
                ax[i].add_artist(leg)
                if i == 2:
                    logm_r = reject_outliers(logm)
                    logm = df.LOGM.isin(logm_r)
                    df_r = df[logm]
                    logt_r = np.log10(tt - df_r['T'])
                    line = np.polyfit(logt_r, logm_r, 1)
                    line_f = np.poly1d(line)
                    ax[i].plot(logt, line_f(logt), color='k')
                    ax[i].legend(['Slope wiht outliers removed is %.2f' %line[0]], loc=2)
        ax[i].grid(b=True)               
    plt.show()

#plot specific heat per site vs. T for different values of N
def C_3D(data):
    color = ['red', 'blue', 'green']
    marker = ['s', '*', 'o']
    c = []
    line = []
    t = []
    for i, d in enumerate(data):
        df = pd.read_csv(d)
        TC = 4.3
        width = .5
        dfn = df[(df['T'] < TC + width) & (df['T'] > TC - width)]
        maxc = max(dfn['C'])
        maxt = dfn[dfn['C'] == maxc]['T']
        T = df['T']
        C = df['C']
        l = plt.scatter(T, C, c=color[i], marker=marker[i])
        c.append(maxc)
        line.append(l)
        t.append(maxt)
    plt.axvline(x=TC-width)
    plt.axvline(x=TC+width)
    plt.xlabel('T')
    plt.ylabel('Specific Heat per site C(T)/N')
    plt.legend([line[0], line[1], line[2]], ['N = 5 with $C_{max}$ = %.2f, and $T_{max}$ = %.2f' %(c[0], t[0]), 'N = 10 with $C_{max}$ = %.2f, and $T_{max}$ = %.2f' %(c[1], t[1]), 'N = 25 with $C_{max}$ = %.2f, and $T_{max}$ = %.2f' %(c[2], t[2])], loc=2)
    plt.grid(b=True)
    plt.show()

#plots the spin orientation of the spin-field
def plot_3d_spins(field):
    plt.figure()
    ax = plt.axes(projection='3d')
    N, M, H = field.shape
    for i in range(N):
        for j in range(M):
            for k in range(H):
                if field[i,j,k] == 1:
                    ax.scatter(i, j, k, c='blue')
                else:
                    ax.scatter(i, j, k, c='red')
    plt.show()

