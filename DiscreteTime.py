import numpy as np
import matplotlib.pyplot as plt

class Signal:
    '''Model a discrete time signal'''
    def __init__(self, dv, start, end):
        '''Create an instance of Signal'''
        self.dv = np.array(dv)
        self.iv = np.arange(start, end + 1)
    
    def plot(self, ylabel="x[n]", title="Graph"):
        '''Make a stem plot fo the signal'''
        plt.stem(self.iv, self.dv)
        plt.xticks(self.iv)
        plt.xlabel("n")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def scale(self, k):
        '''Scale a signal by a factor of k'''
        self.dv *= k

    def flip(self):
        '''Flips a signal'''
        self.dv = np.flip(self.dv)
        self.iv = -1 * np.flip(self.iv)

    def shift(self, nx):
        '''Shifts a signal by some entered amount'''
        self.dv = np.pad(self.dv, (abs(nx) if nx >= 0 else 0,abs(nx) if nx <= 0 else 0))
        self.iv = np.arange((self.iv[0] if nx >= 0 else (self.iv[0] - abs(nx))),((self.iv[-1] + abs(nx) + 1) if nx >= 0 else self.iv[-1] + 1))

    def decimate(self, D):
        '''Decimates a signal by some entered amount'''
        self.dv = np.compress((self.iv % D) == 0, self.dv)
        self.iv = np.compress((self.iv % D) == 0, self.iv) / D

    def expand(self, U):
        '''Expands a signal by some entered amount'''
        self.dv = np.zeros(self.dv.size * U - (U - 1))
        self.iv = np.arange(self.iv[0] * U, self.iv[-1] * U + 1, 1)
        self.dv[::U] = self.dv

    #Operator Overload
    def __add__(self, other):
        '''Add two signals'''
        A, B = Signal.mapSignals(self, other)
        return Signal(A.dv + B.dv, A.iv[0], B.iv[-1])

    def __sub__(self, other):
        '''Sub two signals'''
        A, B = Signal.mapSignals(self, other)
        return Signal(A.dv - B.dv, A.iv[0], B.iv[-1])

    def __mul__(self, other):
        '''Mult two signals'''
        A, B = Signal.mapSignals(self, other)
        return Signal(A.dv * B.dv, A.iv[0], B.iv[-1])

    def __eq__(self, other):
        '''Overloads the == operator'''
        A, B = Signal.mapSignals(self, other)
        return np.array_equal(A.dv, B.dv)

    @staticmethod
    def mapSignals(A, B): # Change to map signals
        '''Align two signals to the same independent variable'''
        iv = np.arange(np.min([A.iv[0], B.iv[0]]), np.max([A.iv[-1], B.iv[-1]]) + 1)
        dvA = np.pad(A.dv, (0 if A.iv[0] <= iv[0] else A.iv[0] - iv[0], 0 if A.iv[-1] >= iv[-1] else iv[-1] - A.iv[-1]))
        dvB = np.pad(B.dv, (0 if B.iv[0] <= iv[0] else B.iv[0] - iv[0], 0 if B.iv[-1] >= iv[-1] else iv[-1] - B.iv[-1]))
        return Signal(dvA, iv[0], iv[-1]), Signal(dvB, iv[0], iv[-1])

    @staticmethod
    def convolve(E, F):
        pass
        '''Convolves 2 signals together from input arguments'''

class Impulse(Signal):
    '''Model the Unit Impulse Sequence'''
    def __init__(self, start, end):
        '''Create an instance of an impulse'''
        self.iv = np.arange(start, end + 1)
        self.dv = 1 * (self.iv == 0) 

class Step(Signal):
    '''Model the Unit Step Sequence'''
    def __init__(self, start, end):
        '''Create an instance of a step'''
        self.iv = np.arange(start, end + 1)
        self.dv = 1 * (self.iv >= 0)
        
class PowerLaw(Signal):
    '''Model the Power Law Sequence'''
    def __init__(self, start, end, A=1, alpha=1):
        '''Create an instance of a PowerLaw'''
        self.iv = np.arange(start, end + 1)
        self.dv = A * (alpha ** self.iv)

class Pulse(Signal):
    '''Model the pulse sequence'''
    def __init__(self, start, end, N = 1):
        '''Create an instance of a Pulse'''
        self.iv = np.arange(start, end + 1)
        step = 1 * (self.iv >= 0)
        sample = -1 * (self.iv >= N)
        self.dv = step + sample

class Sinusoid(Signal):
    '''Model the sinusoid sequence'''
    def __init__(self, start, end, A, omega, phi):
        '''Create an instance of a Sinusoid'''
        self.iv = np.arange(start, end + 1)
        self.dv = A * np.cos((omega * self.iv) + phi)

if __name__ == "__main__":
    a = Pulse(-2,8,4)
    a.plot("a[n]", "Pulse")
    b = Sinusoid(-5,15,1,3*np.pi/4,0)
    b.plot("b[n]","Sinusoid")