import numpy as np 
import pandas as pd 
import os
import re

#Plotting dependancies
import matplotlib.pyplot as plt 
from matplotlib import colors 
from matplotlib.legend import Legend


class TwoDModel:
    '''
    This is a class that creates a random spin field of size nSites x nSites. 
    
    Attributes:
        J_value (float/int) : The interaction value between neighbouring particles in the spin field, 
            a positive value represents a Feromagnetic material, while a negative value would be an 
            antiferomagnetic material
    '''

    
    def __init__(self, J_value=1):
        '''
        The constructor for TwoDModel class. 

        Parameters:
            J_value (float/int) : The interaction value between neighbouring particles in the spin field, 
                a positive value represents a Fermomagnetic material, while a negative value would be an
                antiferomagnetic material
            nEquil (int) : The number of Ising Updates performed on the spin field to bring it to equillibrium
            nSteps (int) : The number of Ising Updates performed after equilibration to caluculate an average
        '''
        self.J = J_value
        self.nEquil = 2400
        self.nSteps = 1240
        
    def RandomField(self, nSites):
        '''
        Creates a 2D Random Spin Field with values of -1 or 1 for each site (spin-up or spin-down)

        Parameters :
            nSites (int) : The size of the Random Spin Field lattice, the shape is nSites x nSites
        '''
        random_field = np.random.choice([-1, 1], size=(nSites, nSites))
        return random_field
    
    def SpinNeighbours(self, field, x, y):
        '''
        Helper function to find the total spin of the neighbouring spin sites to a given x, y coordinate

        Parameters:
            field (numpy array) : The spin field we are trying to find the properties of 
            x, y (int) : a pair of coordinates that define a single particle in the field
        '''
        N, M  = field.shape
        s = field[x, y]
        nb = field[(x+1)%N, y] + field[x, (y+1)%M] +\
            field[(x-1)%N, y] + field[x, (y-1)%M]
        return s, nb


    
    def ising_update(self, field,  T):
        '''
        Main function to equilibrate the spin field at a given temperature. Uses the Monte-Carlo random
        sampling method. Performs N^M ising updates to the field for each instance of ising_update. 
        With each iteration the method finds a random spin site and finds the energy of the lattice 
        portion and if it is less than 0 or if a random sample is less than exp(-dE/T) the spin is 
        flipped. This produces a lowest energy state after equilibration is complete.

        Parameters :
            field (numpy array) : The spin field we are trying to equilibrate
            T (float) : Temperature of the surrounding environment with which the field is equilibrated
                        with
        '''
        N, M = field.shape
        J = self.J
        snb = self.SpinNeighbours
        for _ in range(N):
            for _ in range(M):
                x = np.random.randint(0, N)
                y = np.random.randint(0, M)
                spin, nb = snb(field, x, y)
                dE = 2*J*spin*nb
                if dE < 0:
                    spin *= -1
                elif np.random.random_sample() < np.exp(-dE/T):
                    spin *= -1
                field[x,y] = spin
        return field

    def equipbrate_field(self, field, T):
        '''
        Equilibrates the spin field for a given temperature utilizing the ising_update method

        Paramters :
            field (numpy array) : The spin field we are trying to equilibrate
            T (flaot) : Temperature of the surrounding enviroment wiht which the field is equilibrated 
                        with
        '''
        nEquil = self.nEquil
        ising = self.ising_update
        for _ in range(0, nEquil):
            field = ising(field, T)
        return field

    def energy(self, field):
        '''
        Calculates the total energy of the spin field, does this by finding the total spinneighbour
        values for every site in the field adn then adding them together and averaging over the duplicated
        sites.

        Parameters :
            field (numpy array) : The spin field we are finding the total energy of
        '''
        N, M = field.shape
        energy = 0
        J = self.J
        snb = self.SpinNeighbours
        for i in range(N):
            for j in range(M):
                s, nb = snb(field, i, j)
                energy += -J*nb*s
        return energy/4.

    
    def mag(self, field):
        '''
        Calculated the total magnetization of the spin field, does this by finding a summation of 
        the spins in the field and taking the absolute value.

        Parameters :
            field (numpy array) : The spin field we are finding the total magnetization of
        '''
        mag = np.sum(field)
        return abs(mag)

    def data_to_csv(self, nSites):
        '''
        Exports ['T','E','M','C','X','LOGM','U'] values to a csv file, named for the number of sites,
        after equilibration of the field and an averaging over nSteps number of values.

        Parameters :
            nSites (list of ints) : list of integer values to use a the side lengths of the square
                        lattices 
        '''
        rfield = self.RandomField
        ising = self.ising_update
        en = self.energy
        mg = self.mag
        start = 1.0
        stop = 3.5
        step = .01
        nt = int((stop-start)/step)
        nEquil = self.nEquil
        nSteps = self.nSteps

        T = np.linspace(start, stop, nt)
        n1t = list(map(lambda x: 1.0/(nSteps*x*x), nSites))
        n2t = list(map(lambda x: 1.0/(nSteps*nSteps*x*x), nSites))
        E, M, C, X, LOGM, U = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
        for k in range(len(nSites)):
            N = nSites[k]
            n1, n2= n1t[k], n2t[k]
            print(f'Strating equilibration for nSites = {nSites[k]}')
            for tt in range(nt):
                E1 = M1 = E2 = M2 = M4 = 0.0
                iT = 1.0/(T[tt]); iT2 = iT*iT
                field = rfield(N)

                for _ in range(nEquil):
                    field = ising(field, T[tt])

                for _ in range(nSteps):
                    field = ising(field, T[tt])
                    Ene = en(field)
                    Mag = mg(field)

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
            df_rejected = self.reject_outliers(df)
            
            print(f'Staring exporting of data for nSites = {nSites[k]}')
            df.to_csv(f'{nSites[k]}_2D.csv', index=False)
            df_rejected.to_csv(f'{nSites[k]}_2D_rejected.csv', index=False)
            
        print("All Done!")

    def reject_outliers(self, df):
        '''
        Rejects outliers from the csv data, the outliers are defined as points with a Magnetization
        value that is greateer than the next value by .00002. This has been chosen because the
        Magnetization is generally supposed to trend down at all points, so if the difference between 
        the current point and the next point is negative we know the point is trending the wrong
        direction and can be rejected.

        Parameters :
            df (pandas data frame) : DataFrame from the csv file data,
                        use df = pd.read_csv(filename, header=0)
        '''
        df_rejected = pd.DataFrame(columns=['T','E','M','C','X','LOGM','U'])
        for index, row in df.iterrows():
            if index == len(df.index) - 1:
                df_rejected = df_rejected.append(df.iloc[index,:])
                return df_rejected
            diff = row['M'] - df.M.iloc[index + 1]
            if diff < -0.0002:
                print('rejected point', diff)
            else:
                df_rejected = df_rejected.append(df.iloc[index,:])
    
    def SpinFieldToFile(self, field):
        '''
        Saves the spin field values to a csv file for plotting.

        Parameters :
            field (numpy array) : Field that you are saving with the filename, 
                    SpinFieldValues_size_2D.csv
        '''
        size = field.shape[0]
        filename = f'SpinFieldValues_{size}_2D.csv'
        np.savetxt(filename, field, delimiter=',')
        print(f'Field saved to {filename}')

class TwoDPlotting:
    '''
    Class used for plotting the attributes of the spin fields from the csv file data.
    Initilization finds all the csv files in the path and saves them to self.data

    Parameters :
        None
    '''
    def __init__(self):
        data = []
        nSites = []
        csv_regex = re.compile(r'(.csv)')
        spinfieldregex = re.compile(r'((?=SpinFieldValues))')
        with os.scandir() as entries:
            for entry in entries:
                if csv_regex.search(entry.name) and spinfieldregex.search(entry.name) == None:
                    data.append(entry.name)
                    N = re.search(r'(\d{1,2})', entry.name)[0]
                    nSites.append(N)
        self.data = data
        self.nSites = nSites
        self.color = ['blue', 'green','red','cyan','magenta','yellow','black']
        self.marker = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', \
            '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', \
                '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    def CalcTMax(self):
        '''
        Helper function to find the temperature of the Maximum Chi^2 value.

        Parameters :
            None
        '''
        data = self.data 
        t = []
        for _, d in enumerate(data):
            df = pd.read_csv(d)
            dfn = df[(df['T'] < 2.5) & (df['T'] > 2.0)]
            maxx = max(dfn['X'])
            maxt = float(dfn[dfn['X'] == maxx]['T'])
            t.append(maxt)
        TMax = round(sum(t)/len(t), 2)
        return TMax
       
    def mag_2D(self):
        '''
        Plots the magnetization versus temperature for all files in the self.data variable

        Parameters :
            None
        '''
        data = self.data
        color = self.color
        marker = self.marker
        for i, d in enumerate(data):
            df = pd.read_csv(d)
            T = df['T']
            M = df['M']
            plt.scatter(T, M, c=color[i%len(self.color)], marker=marker[i])
        plt.legend([f'N ={self.nSites[k]}' for k in range(len(data))], loc=3)
        plt.xlabel('T')
        plt.ylabel('$|<M>|/N$')
        plt.grid(b=True)
        plt.show()

    def energy_2D(self):
        '''
        Plots the energy versus temperature values for all the files in the self.data variable

        Parameters :
            None
        '''
        data = self.data
        color = self.color
        marker = self.marker
        for i, d in enumerate(data):
            df = pd.read_csv(d)
            T = df['T']
            E = df['E']
            plt.scatter(T, E, color=color[i%len(self.color)], marker=marker[i])
        plt.legend([f'N = {self.nSites[k]}' for k in range(len(data))], loc=2)
        plt.xlabel('T')
        plt.ylabel('Energy')
        plt.grid(b=True)
        plt.show()

    def chi_2D(self):
        '''
        Plots the Chi^2 versus temperature values for all the fiels in the self.data variable.
        Finds and plots the temperature of the maximum Chi^2 value using the CalcTMax() method.

        Parameters :
            None
        '''
        data = self.data
        color = self.color
        marker = self.marker
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
            l = plt.scatter(T, chi, color=color[i%len(self.color )], marker=marker[i])
            line.append(l)
        plt.legend([line[k] for k in range(len(line))], ['N = %s with $chi^2_{max}$ = %.2f at $T_{max}$ = %.2f' %(self.nSites[k], c[k], t[k]) for k in range(len(c))])
        plt.xlabel('T')
        plt.ylabel('Magnetic Susceptiblity')
        plt.axvline(x=TC-width)
        plt.axvline(x=TC+width)
        plt.grid(b=True)
        plt.xlim(1.9, 2.9)
        plt.show()

    def logmag_2D(self):
        '''
        Plots the log values of magnetism versus temperature as well as the lines that divide the
        sections where we search for the maxium logmag value
        
        Parameters :
            None
        '''
        data = self.data
        color = self.color
        _, ax = plt.subplots(1, len(data), sharey=True)
        TMAX = self.CalcTMax()
        T_under = np.linspace(2.1, TMAX, 4)
        T_over = np.linspace(TMAX, 2.8, 4)
        T = np.sort(np.append(T_under, T_over))
        for i, d in enumerate(data):
            for j, tt in enumerate(T):
                df = pd.read_csv(d)
                df = df[df['T'] < tt]
                logt = np.log10(tt - df['T'])
                logm = df['LOGM']
                ax[i].scatter(logt, logm, color=color[j%len(color)])
                if tt == TMAX:
                    line = np.polyfit(logt, logm, 1)
                    line_f = np.poly1d(line)
                    line1 = ax[i].plot(logt, line_f(logt), color=color[j], label='Original')
                    ax[i].legend(line1, ['T is %s and Slope is %s' %(tt, line[0])], loc=3)

                    
                    logt_r = np.log10(tt - df['T'])
                    line = np.polyfit(logt_r, logm, 1)
                    line_f = np.poly1d(line)
                    line2 = ax[i].plot(logt, line_f(logt), color='black', label='Outliers Removed')
                    leg = Legend(ax[i], line2, ['T is %s and Slope is %s' %(tt, line[0])], loc=2)
                    ax[i].add_artist(leg)
            ax[i].grid(b=True)               
        plt.show()

    def C_2D(self):
        '''
        Plots the specific heat versus T from the csv files in the self.data variable

        Parameters :
            None
        '''
        data = self.data
        color = self.color
        marker = self.marker
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
            l = plt.scatter(T, C, c=color[i%len(self.color)], marker=marker[i])
            c.append(maxc)
            line.append(l)
            t.append(maxt)
        plt.axvline(x=TC-width)
        plt.axvline(x=TC+width)
        plt.xlabel('T')
        plt.ylabel('Specific Heat per site C(T)/N')
        plt.legend([line[k] for k in range(len(line))], ['N = %s with $C_{max}$ = %.2f, and $T_{max}$ = %.2f' %(self.nSites[i], c[i], t[i]) for i in range(len(c))], loc=2)
        plt.grid(b=True)
        plt.show()

    def PlotAll(self):
        '''
        Runs all of the methods in TwoDPlotting().

        Parameter :
            None
        '''
        self.mag_2D()
        self.energy_2D()
        self.chi_2D()
        self.logmag_2D()
        self.C_2D()
        
class PlotSpinField:
    '''
    Helper class for visualizing the spin fields produced with the TwoDModel class.

    Parameters :
        None
    '''
    #arrows that show a visulaization of a small spin-field
    def arrows(self, field):
        '''
        Plots the spin fields represented with up and down arrows for spin up and spin down
        values reprectivley.

        Parameters :
            field (numpy array) : Spin field, can get from saved field by using from numpy import genfromtxt
                    field = genfromtxt(filename, delimiter=',')
        '''
        x, y = field.shape
        for i in range(x):
            for j in range(y):
                color = 'black'
                spin = field[i][j]
                if spin == -1 :
                    j = j + 0.65
                    color = 'red'
                plt.quiver(i, j, 0, spin, color=color)
        plt.ylim(0, 10)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    #plots the spins of the spin-field as black and white
    #need to create another method to output field values at differnt T values
    def plot(self, field):
        '''
        Plots the spin fields represented with black and white boxes for each particle, black is spin
        up and white is spin down.

        Parameters :
            field (numpy array) : Spin field, can get from saved field by using from numpy import genfromtxt
                    field = genfromtxt(filename, delimiter=',')
        '''
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
