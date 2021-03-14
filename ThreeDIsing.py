import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.npyio import genfromtxt 
import pandas as pd 


class ThreeDModel :

    def __init__(self, J_value=1):
        self.J = J_value
        self.nEquil = 2400
        self.nSteps = 1240


    def SpinNeighbours(self, field, x, y, z):
        N, M, H = field.shape
        spin = field[x,y,z]
        total = field[(x+1)%N,y,z] + field[(x-1)%N,y,z] + field[x,(y+1)%M,z] + \
                field[x,(y-1)%M,z] + field[x,y,(z+1)%H] + field[x,y,(z-1)%H] 
        return spin, total

    def RandomField(self, nSites):
        random_field = np.random.choice([-1, 1], size=(nSites, nSites, nSites))
        return random_field

    def ising_update(self, field, T):
        N, M, H = field.shape
        for _ in range(N):
            for _ in range(M):
                for _ in range(H):
                    x = np.random.randint(N)
                    y = np.random.randint(M)
                    z = np.random.randint(H)
                    spin, total = self.SpinNeighbours(field, x, y, z)
                    dE = 2*self.J*spin*total
                    if dE < 0:
                        spin *= -1
                    elif np.random.random_sample() < np.exp(-dE/T):
                        spin *= -1
                    field[x,y,z] = spin
        return field

    def energy(self, field):
        N, M, H = field.shape
        energy = 0
        for i in range(N):
            for j in range(M):
                for k in range(H):
                    spin, total = self.SpinNeighbours(field, i, j, k)
                    energy += -self.J*total*spin
        return energy/6

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

    def mag(self, field):
        mag = np.sum(field)
        return abs(mag)

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
        start = 2.0
        stop = 7.5
        step = .5
        nt = int((stop-start)/step)
        nSteps = self.nSteps

        T = np.linspace(start, stop, nt)
        n1t = list(map(lambda x: 1.0/(nSteps*x*x*x), nSites))
        n2t = list(map(lambda x: 1.0/(nSteps*nSteps*x*x*x), nSites))
        E, M, C, X, LOGM, U = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
        for k in range(len(nSites)):
            N = nSites[k]
            vol = N*N*N
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
                U[tt] = 1 - (M4/3*(M2*M2)*vol)
                LOGM[tt] = np.log10(n1*M1) 

            self.SpinFieldToFile(field)
            data = np.column_stack((T,E,M,C,X,LOGM,U))
            df = pd.DataFrame(data, columns=['T','E','M','C','X','LOGM','U'])
            df_rejected = self.reject_outliers(df)
            
            print(f'Staring exporting of data for nSites = {nSites[k]}')
            df.to_csv(f'{nSites[k]}_3D.csv', index=False)
            df_rejected.to_csv(f'{nSites[k]}_3D_rejected.csv', index=False)

        print("All Done!")
        

    def SpinFieldToFile(self, field):
        '''
        Saves the spin field values to a csv file for plotting.

        Parameters :
            field (numpy array) : Field that you are saving with the filename, 
                    SpinFieldValues_size_2D.csv
        '''
        size = field.shape[0]
        field_reshaped = field.reshape(size, -1)
        filename = f'SpinFieldValues_{size}_3D.csv'
        np.savetxt(filename, field_reshaped, delimiter=',')
        print(f'Field saved to {filename}')
        
class ThreeDSpinPlotting :
    '''
    Helper class for visualizing the spin fields produced with the TwoDModel class.

    Parameters :
        None
    '''
    def __init__(self, filename) -> None:
        size = int(filename.split('_')[1])
        print(size)
        field_reshaped = genfromtxt(filename, delimiter=',')
        self.field = field_reshaped.reshape(
            field_reshaped.shape[0], field_reshaped.shape[1] // size, size
        )

    def plot_3d_spins(self):
        '''
        Plots the spin fields represented with blue and red spheres for each particle, blue is spin up
        and red is spin down.

        Parameters :
            field (numpy array) : Spin field, can get from saved field by using from numpy import genfromtxt
                    field = genfromtxt(filename, delimiter=',')
        '''
        plt.figure()
        ax = plt.axes(projection='3d')
        N, M, H = self.field.shape
        for i in range(N):
            for j in range(M):
                for k in range(H):
                    if self.field[i,j,k] == 1:
                        ax.scatter(i, j, k, c='blue')
                    else:
                        ax.scatter(i, j, k, c='red')
        plt.show()
