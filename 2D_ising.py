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

    def equilibrate_field(self, field, T):
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

                field = self.equilibrate_field(field, T[tt])

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
        
class TwoDSpinPlotting:
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
