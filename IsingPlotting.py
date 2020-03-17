import numpy as np 
import pandas as pd 
import os
import re

import matplotlib.pyplot as plt 
from matplotlib import colors 
from matplotlib.legend import Legend

class Plotting :
    '''
    Class used for plotting the attributes of the spin fields from the csv file data.
    Initilization finds all the csv files in the path and saves them to self.data

    Parameters :
        dim (int) : 2 or 3, dimmension of data you want to plot
    '''
    def __init__(self, dim):
        data = []
        nSites = []
        dim_string = f'{dim}D'
        csv_regex = re.compile(f'({dim_string}(_rejected)*.csv)')
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
        if dim == 2:
            self.TC = 2.35
        elif dim == 3:
            self.TC = 4.25
        else:
            print('invalid dimension chosen')
            exit()
    
    def CalcTMax(self):
        '''
        Helper function to find the temperature of the Maximum Chi^2 value.

        Parameters :
            None
        '''
        data = self.data 
        t = []
        TC = self.TC
        for _, d in enumerate(data):
            df = pd.read_csv(d)
            dfn = df[(df['T'] < TC + 1) & (df['T'] > TC - 1)]
            maxx = max(dfn['X'])
            maxt = float(dfn[dfn['X'] == maxx]['T'])
            t.append(maxt)
        TMax = round(sum(t)/len(t), 2)
        return TMax
       
    def Mag(self):
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

    def Energy(self):
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

    def Chi(self):
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
        TC = self.TC
        maxx_list = []
        for i, d in enumerate(data):
            df = pd.read_csv(d)
            T = df['T']
            chi = df['X']
            width = 0.5
            dfn = df[(df['T'] < TC + width) & (df['T'] > TC - width)]
            maxx = max(dfn['X'])
            maxx_list.append(maxx)
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
        plt.xlim(TC - TC/2, TC + TC/2)
        plt.ylim(0, max(maxx_list))
        plt.show()

    def LogMag(self):
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
                    line1 = ax[i].plot(logt, line_f(logt), color=color[j%len(color)], label='Original')
                    ax[i].legend(line1, ['T is %s and Slope is %s' %(tt, line[0])], loc=3)

                    
                    logt_r = np.log10(tt - df['T'])
                    line = np.polyfit(logt_r, logm, 1)
                    line_f = np.poly1d(line)
                    line2 = ax[i].plot(logt, line_f(logt), color='black', label='Outliers Removed')
                    leg = Legend(ax[i], line2, ['T is %s and Slope is %s' %(tt, line[0])], loc=2)
                    ax[i].add_artist(leg)
            ax[i].grid(b=True)               
        plt.show()

    def C(self):
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
        TC = self.TC
        for i, d in enumerate(data):
            df = pd.read_csv(d)
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
        Runs all of the methods in ThreeDPlotting().

        Parameter :
            None
        '''
        self.Mag()
        self.Energy()
        self.Chi()
        self.LogMag()
        self.C()