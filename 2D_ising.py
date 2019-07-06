import numpy as np 
from random import randint
import matplotlib.pyplot as plt 
from matplotlib import colors 
from numpy.random import rand
import pandas as pd 
'''This program has a multitude of methods to numerically solve the 2-dimmensional ising
model probelm. Specifically there are methods for manipulating a spin-field, other methods 
for calculating values from this spin-field for different temperatures and spin-field sizes, 
and methods to plot these values obtained for different conditions of the spin-field.'''

#nearest neighbour interaction strength
J = 1

#defines a randomly oriented spin-field
def init(nSites):
    return np.random.choice([-1, 1], size=(nSites, nSites))

#does M^N monte-carlo updates to the spin-field "field"
def ising_update(field,  T):
    N, M = field.shape
    for _ in range(N):
        for _ in range(M):
            x = np.random.randint(0, N)
            y = np.random.randint(0, M)
            spin = field[x,y]
            nb = field[(x+1)%N, y] + field[x, (y+1)%M] +\
                field[(x-1)%N, y] + field[x, (y-1)%M]
            dE = 2*J*spin*nb
            if dE < 0:
                spin *= -1
            elif rand() < np.exp(-dE/T):
                spin *= -1
            field[x,y] = spin
    return field

#calculates the energy of the spin-field "field"
def energy(field):
    energy = 0
    N, M = field.shape
    for i in range(N):
        for j in range(M):
            s = field[i, j]
            nb = field[(i+1)%N, j] + field[i,(j+1)%M] + field[(i-1)%N,j] + field[i,(j-1)%M]
            energy += -J*nb*s
    return energy/4.

#calculates the magnetization of the spin-field "field"
def mag(field):
    mag = np.sum(field)
    return abs(mag)

#exports ['T','E','M','C','X','LOGM','U'] values to csv files named 
#[20_2D.csv, 40_2D.csv, 128_2D.csv] indicating the side length of the lattice
#and that it is in 2D, so a square
def data():
    nSites = [20,40,128]
    for k in range(len(nSites)):
        start = 1.0
        stop = 3.5
        step = .05
        nt = int((stop-start)/step)
        N = nSites[k] 
        nEquil = 2400
        nSteps = 1240
        T = np.linspace(start, stop, nt)
        E, M, C, X, LOGM, U = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
        n1, n2= 1.0/(nSteps*N*N), 1.0/(nSteps*nSteps*N*N)
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
                M4 = M4 + pow(Mag, 4)*n1

            E[tt] = n1*E1
            M[tt] = n1*M1
            C[tt] = (E2 - n2*E1*E1)*iT2
            X[tt] = (M2 - n2*M1*M1)*iT
            U[tt] = 1 - (M4/3*(M2*M2)*(N*N))
            LOGM[tt] = np.log10(n1*M1) 
        data = np.column_stack((T,E,M,C,X,LOGM,U))
        df = pd.DataFrame(data, columns=['T','E','M','C','X','LOGM','U'])
        if k == 0:
            df.to_csv('20_2D.csv', index=False)
        elif k == 1:
            df.to_csv('40_2D.csv', index=False)
        else:
            df.to_csv('128_2D.csv', index=False)
    print("All Done!")

def mag_2D(data):
    color = ['red', 'blue', 'green']
    marker = ['s', '*', 'o']
    for i, d in enumerate(data):
        df = pd.read_csv(d)
        T = df['T']
        M = df['M']
        plt.scatter(T, M, c=color[i], marker=marker[i])
    plt.legend(['N = 20', 'N = 40', 'N = 128'], loc=3)
    plt.xlabel('T')
    plt.ylabel('$|<M>|/N$')
    plt.grid(b=True)
    plt.show()

def energy_2D(data):
    color = ['red', 'blue', 'green']
    marker = ['s', '*', 'o']
    for i, d in enumerate(data):
        df = pd.read_csv(d)
        T = df['T']
        E = df['E']
        plt.scatter(T, E, color=color[i], marker=marker[i])
    plt.legend(['N = 20', 'N = 40', 'N = 128'], loc=2)
    plt.xlabel('T')
    plt.ylabel('Energy')
    plt.grid(b=True)
    plt.show()

def chi_2D(data):
    color = ['red', 'blue', 'green']
    marker = ['s', '*', 'o']
    c = []
    t = []
    line = []
    for i, d in enumerate(data):
        df = pd.read_csv(d)
        T = df['T']
        chi = df['X']
        TC = 2.27
        width = 0.25
        dfn = df[(df['T'] < TC + width) & (df['T'] > TC - width)]
        maxx = max(dfn['X'])
        maxt = dfn[dfn['X'] == maxx]['T']
        c.append(maxx)
        t.append(maxt)
        l = plt.scatter(T, chi, color=color[i], marker=marker[i])
        line.append(l)
    plt.legend([line[0], line[1], line[2]], ['N = 20 with $C_{max}$ = %.2f at $T_{max}$ = %.2f' %(c[0], t[0]), 'N = 40 with $C_{max}$ = %.2f at $T_{max}$ = %.2f' %(c[1], t[1]), 'N = 128 with $C_{max}$ = %.2f at $T_{max}$ = %.2f' %(c[2], t[2])], loc=2)
    plt.xlabel('T')
    plt.ylabel('Magnetic Susceptiblity')
    plt.axvline(x=TC-width)
    plt.axvline(x=TC+width)
    plt.grid(b=True)
    plt.xlim(1.9, 2.9)
    plt.show()

def reject_outliers(data, m=0.5):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def logmag_2D(data):
    color = ['red', 'blue', 'green', 'orange', 'purple']
    f, ax = plt.subplots(1, 3, sharey=True)
    T = np.linspace(2.1, 2.5, 5)
    for i, d in enumerate(data):
        for j, tt in enumerate(T):
            df = pd.read_csv(d)
            df = df[df['T'] < tt]
            logt = np.log10(tt - df['T'])
            logm = df['LOGM']
            ax[i].scatter(logt, logm, color=color[j])
            if j == 2:
                line = np.polyfit(logt, logm, 1)
                line_f = np.poly1d(line)
                ax[i].plot(logt, line_f(logt), color=color[j])
                leg = ax[i].legend(['T = %.2f with slope %.2f' %(tt, line[0])], loc=3)
                ax[i].add_artist(leg)
                if i == 1 or i == 2:
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

def C_2D(data):
    color = ['red', 'blue', 'green']
    marker = ['s', '*', 'o']
    c = []
    line = []
    t = []
    for i, d in enumerate(data):
        df = pd.read_csv(d)
        TC = 2.27
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
    plt.legend([line[0], line[1], line[2]], ['N = 20 with $C_{max}$ = %.2f, and $T_{max}$ = %.2f' %(c[0], t[0]), 'N = 40 with $C_{max}$ = %.2f, and $T_{max}$ = %.2f' %(c[1], t[1]), 'N = 128 with $C_{max}$ = %.2f, and $T_{max}$ = %.2f' %(c[2], t[2])], loc=2)
    plt.grid(b=True)
    plt.show()

def arrows():
    x = np.linspace(0, 1, num=4)
    y = np.linspace(0, 1, num=4)
    for i, xpos in enumerate(x):
        for j, ypos in enumerate(y):
            if i > 0 & i < len(x):
                if j > 0 & j< len(y):
                    x_dir = 0
                    y_dir = 1
                    plt.quiver(xpos, ypos, x_dir, y_dir)
    plt.ylim(0.3, 1.1)
    plt.xticks([])
    plt.yticks([])
    plt.show()

#plots the spins of the spin-field as black and white
#need to create another method to output field values at differnt T values
def plot(field):
    
    cmap = colors.ListedColormap(['white', 'black'])
    bounds = [-1, 0 , 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()
    ax = fig.subplots()
    ax.imshow(field, cmap=cmap, norm=norm)

    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(field.shape[0] + 1, 1))
    ax.set_yticks(np.arange(field.shape[1] + 1, 1))
    
    plt.show()

