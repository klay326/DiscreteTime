from venv import create
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
        return self

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
    def convolve(A, B):
        '''Convolves 2 signals together from input arguments'''
        A, B = Signal.mapSignals(A, B)
        y = Signal(np.zeros(A.dv.size), A.iv[0], B.iv[-1])
        for x, k in zip(A.dv, A.iv):
            B_temp = Signal(B.dv, B.iv[0], B.iv[-1])
            B_temp.scale(x)
            B_temp.shift(k)
            y = y + B_temp
        return y


class Impulse(Signal):
    '''Model the Unit Impulse Sequence'''
    def __init__(self, start, end):
        '''Create an instance of an impulse'''
        self.model = lambda n : 1 * (n == 0)
        self.iv = np.arange(start, end + 1)
        self.dv = self.model(self.iv)

class Step(Signal):
    '''Model the Unit Step Sequence'''
    def __init__(self, start, end):
        '''Create an instance of a step'''
        self.model = lambda n : 1 * (n >= 0)
        self.iv = np.arange(start, end + 1)
        self.dv = self.model(self.iv)
        
class PowerLaw(Signal):
    '''Model the Power Law Sequence'''
    def __init__(self, start, end, A=1, alpha=1):
        '''Create an instance of a PowerLaw'''
        self.model = lambda n : 1 * (A * (alpha ** n))
        self.iv = np.arange(start, end + 1)
        self.dv = self.model(self.iv)

class Pulse(Signal):
    '''Model the pulse sequence'''
    def __init__(self, start, end, N):
        '''Create an instance of a Pulse'''
        self.iv = np.arange(start, end + 1)
        step = 1 * (self.iv >= 0)
        sample = -1 * (self.iv >= N)
        self.dv = step + sample

class Pulsealt(Signal):
    '''Model the pulse sequence'''
    def __init__(self, start, end, N):
        '''Create an instance of a Pulse'''
        self.model = lambda n : 1 * np.logical_and(n >= 0, n < N)
        self.iv = np.arange(start, end + 1)
        self.dv = self.model(self.iv)

class Sinusoid(Signal):
    '''Model the sinusoid sequence'''
    def __init__(self, start, end, A=1, omega = 2*np.pi, phi=0):
        '''Create an instance of a Sinusoid'''
        self.model = lambda n : A * np.cos((omega * n) + phi)
        self.iv = np.arange(start, end + 1)
        self.dv = self.model(self.iv)

class System:
    '''Model a Discrete-Time System'''
    def __init__(self, h):
        '''Create an instance of System'''
        self.h = h

    def evaluate(self, x):
        return Signal.convolve(x, self.h)

class ComplexExp(Signal):
    '''Model the complex exponential sequence'''
    def __init__(self, start, end, A=1, omega = 2*np.pi, phi=0):
        '''Create an instance of COmplexExp'''
        self.model = lambda n : A * np.exp(1J * (omega * n + phi))
        self.iv = np.arange(start, end + 1)
        self.dv = self.model(self.iv)

    def plot(self, ylabel="x[n]", title="Complex Exponential", real=True):
        '''Plot the complex exponential sequence'''
        plt.stem(self.iv, np.real(self.dv) if real else np.imag(self.iv))
        plt.xticks(self.iv)
        plt.xlabel("n")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

if __name__ == "__main__":
    x = Signal([1,2,-1], 0, 2)
    h = Signal([1,1,1,1], 0, 3)
    y = Signal.convolve(x, h)
    y.plot("y[n]", "Convolution")
