# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-12-02 15:06:17
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2024-12-02 16:38:59
import numpy as np
from scipy import signal
import pylab as plt
import os, sys, pickle


class solveTD:
    def __init__(self, filename):
        self.read_file(filename)
        self.isolate_segments()
        self.plot()

    def shift(self, f, shift_time):
        offset = int(shift_time // self.dt)
        print("shifting by:", offset)
        return np.roll(f, offset)

    def isolate_segments(self):
        RA = (self.lBar + (self.lBar - self.locA)) / self.c0
        RB = (self.lBar + (self.lBar - self.locB)) / self.c0
        delta = 2 * (self.locB - self.locA) / self.c0
        print("RA:", RA)
        print("RB:", RB)

        #
        # SEGMENT 0
        #

        # at B is simply the ascending wave at A
        tCut = RA - delta
        self.eps_A_asc0 = np.where(self.t < RA, self.epsA, 0.0)

        #
        # SEGMENT 1
        #
        i = 1
        # this is the ascending wave at A. It is known initially
        self.eps_A_asc1 = np.where(np.logical_and(RA + (i - 2) * delta <= self.t, self.t < RA + (i - 1) * delta), self.epsA, 0.0) #np.where(T <= RA, eps_A, 0.0)
        # propagate the ascending wave from A to B
        self.eps_B_asc1 = self.shift(self.eps_A_asc1, 0.5 * delta + self.dt)
        # isolate wave at B.
        self.eps_B_1 = np.where(np.logical_and(RB + (i - 1) * delta <= self.t, self.t < RB + i * delta), self.epsB, 0.0)
        # compute the descending wave at B
        self.eps_B_dsc1 = self.eps_B_1 - self.eps_B_asc1

        # total ascending and descending waves at B
        self.eps_B_dsc = self.eps_B_dsc1
        self.eps_B_asc = self.shift(self.eps_A_asc0, 0.50000 * delta )

        # shift B waves to specimen center
        
        self.eps_BS_asc = self.shift(self.eps_B_asc, 2*(self.lBar - self.locB) / self.c0)
        self.eps_BS_dsc = self.shift(self.eps_B_dsc, -5*self.dt)

    def plot(self):
        fig = plt.figure()
        gs = fig.add_gridspec(3, hspace=0)
        axs = gs.subplots(sharex=True, sharey=False)
        fig.suptitle('Sharing both axes')
        axs[0].plot(self.t, self.epsA, label="A")
        axs[0].plot(self.t, self.epsB, label="B")
        axs[0].legend()
        
        #axs[1].plot(self.t, self.eps_A_asc0, "--", label="A_asc0")
        #axs[1].plot(self.t, self.eps_A_asc1, label="A_asc1")
        #axs[1].plot(self.t, self.eps_B_1, label="B_1")
        #axs[1].plot(self.t, self.eps_B_dsc1, label="B_dsc1")
        #axs[1].plot(self.t, self.eps_B_asc1, "r-", label="eps_B_asc1")

        axs[1].plot(self.t, self.eps_B_asc, label="B_asc")
        axs[1].plot(self.t, self.eps_B_dsc, label="B_dsc")
        axs[1].legend()


        #axs[2].plot(self.t, self.eps_BS_asc, label="BS_asc")
        #axs[2].plot(self.t, self.eps_BS_dsc, label="BS_dsc")
        axs[2].plot(self.t, self.eps_BS_asc + self.eps_BS_dsc, label="BS_sum")
        axs[2].plot(self.t, self.epsC, label="C")
        axs[2].legend()

        
        plt.show()

        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()

        


    def read_file(self, filename):

        data = np.genfromtxt(filename)
        self.t = data[:,0]
        self.epsA = data[:,1]
        self.epsB = data[:,2]
        self.epsC = data[:,3]
        self.dt = self.t[1] - self.t[0]
        N = len(self.t)
        print("read file [%s], dt=%g, Nrec=%d " % (filename, self.dt, N))

        self.locA = 1500
        self.locB = 2800
        self.lBar = 3000
        self.c0   = 5091.750772173155

    


solver = solveTD("ABC.dat")