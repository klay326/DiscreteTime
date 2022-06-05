import numpy as np
import matplotlib.pyplot as plt

class Signal:
    '''Model a discrete time signal'''
    def __init__(self, x, start, end):
        '''Create an instance of Signal'''
        self.x = np.array(x)
        self.n = np.arange(start, end + 1)
    
    def plot(self, ylabel="x[n]"):
        '''Make a stem plot fo the signal'''
        plt.stem(self.n, self.x)
        plt.xticks(self.n)
        plt.xlabel("n")
        plt.ylabel(ylabel)
        plt.show()

    def __add__(self, other):
        '''Add two signals'''
        x = self.x + other.x
        n = self.n
        return Signal(x, n[0], n[-1])

    def flip(self):
        '''Flips a signal'''
        xC = self.x
        nC = self.n
        xC = np.flip(xC)
        nC = -1 * np.flip(nC)
        return Signal(xC, nC[0], nC[-1])

    def shift(self, n0):
        '''Shifts a signal by some entered amount'''
        xD = self.x
        nD = self.n
        xD = np.pad(xD, (abs(n0) if n0 >= 0 else 0,abs(n0) if n0 <= 0 else 0))
        nD = np.arange((nD[0] if n0 >= 0 else (nD[0] - abs(n0))),((nD[-1] + abs(n0) + 1) if n0 >= 0 else nD[-1] + 1))
        print(self.n)
        print(nD)

        return Signal(xD, nD[0], nD[-1])

    def decimate(self, D):
        '''Decimates a signal by some entered amount'''
        xE = self.x
        nE = self.n
        xE = np.compress((nE % D) == 0, xE)
        nE = np.compress((nE % D) == 0, nE) / D
        return Signal(xE, nE[0], nE[-1])

    def expand(self, U):
        '''Expands a signal by some entered amount'''
        xF = self.x
        nF = self.n
        xF = np.zeros(xF.size * U - (U - 1))
        nF = np.arange(nF[0] * U, nF[-1] * U + 1, 1)
        xF[::U] = self.x
        return Signal(xF, nF[0], nF[-1])

    @staticmethod
    def matchSignals(A, B):
        '''Align two signals to the same independent variable'''
        xA = A.x
        nA = A.n
        xB = B.x
        nB = B.n

        n = np.arange(np.min([nA[0], nB[0]]), np.max([nA[-1], nB[-1]]) + 1)
        xA = np.pad(xA, (0 if nA[0] <= n[0] else nA[0] - n[0], 0 if nA[-1] >= n[-1] else n[-1] - nA[-1]))
        xB = np.pad(xB, (0 if nB[0] <= n[0] else nB[0] - n[0], 0 if nB[-1] >= n[-1] else n[-1] - nB[-1]))
        return Signal(xA, n[0], n[-1]), Signal(xB, n[0], n[-1])



    if __name__ == "__main__":


