import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import triang
from scipy.fftpack import fft




class week2Sinusoid():
    def __init__(self):
        self.A = .8
        self.f0 = 1000
        self.phi = np.pi/2
        self.fs = 44100

    def realSinusoid(self):
        print("this is real SIN")
        t = np.arange (-.002, .002, 1.0/self.fs)
        # real sin
        x = self.A * np.cos(2 * np.pi * self.f0 * t + self.phi)
        plt.plot(t, x)
        plt.axis([-.002, .002, -.8, .8])
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()

    def complexSinusoid(self):
        print("this is real part in complex SIN")
        N = 1204
        k = 5
        n = np.arange(-N/2, N/2)
        # complex sin
        s = np.exp(1j * 2 * np.pi* k * n / N)
        plt.plot(n, np.real(s))
        plt.axis([-N/2, N/2-1, -1, 1])
        plt.xlabel('n')
        plt.ylabel('amplitude')
        plt.show()

        print("this is imaginary part in complex SIN")
        plt.plot(n, np.imag(s))
        plt.show()

    def test(self):
        self.realSinusoid()
        self.complexSinusoid()




class week2DFT():
    def __init__(self):
        self.N = 64
        self.k0 = 7
        self.x = np.cos(2 * np.pi * self.k0 / self.N * np.arange(self.N))
        self.nv = np.arange(-self.N/2, self.N/2)
        self.kv = np.arange(-self.N/2, self.N/2)

    def DFT(self):
        X = np.array([])
        for k in self.kv:
            s = np.exp(1j * 2 * np.pi * k / self.N * self.nv)
            X = np.append(X, sum(self.x*np.conjugate((s))))
        plt.plot(self.kv, abs(X))
        plt.axis([-self.N/2, self.N/2-1, 0, self.N])
        plt.show()
        return X

    def IDFT(self,X):
        y = np.array([])
        for n in self.nv:
            s = np.exp(1j * 2 * np.pi * n / self.N * self.kv)
            y = np.append(y, 1.0/self.N * sum(X*s))
        plt.plot(self.kv, y)
        plt.axis([-self.N/2, self.N/2-1, -1, 1])
        plt.show()
        return y

    def test(self):
        print("This is orginal signal from complex exponential")
        plt.plot(self.x)
        plt.show()
        print("Show freq spectrum")
        X = self.DFT()
        print("Ok, now we show signal from inverse transform of freq spectrum")
        self.IDFT(X)


class _fft():
    def __init__(self):
        self.x = triang(15)
        #self.X = np.fft.fft(self.x)
        self.X = fft(self.x)
        self.mX = abs(self.X)
        self.pX = np.angle(self.X)

    def showAll(self):
        list = [self.x, self.X, self.mX, self.pX]
        for element in list:
            plt.plot(element)
            plt.show()

    def trishift(self):
        fftbuff = np.zeros(15)
        fftbuff[:8] = self.x[7:]
        fftbuff[8:] = self.x[:7]
        self.x = fftbuff
        self.X = fft(fftbuff)
        self.mX = abs(self.X)
        self.pX = np.angle(self.X)
        self.showAll()

    def test(self):
        self.trishift()


class week4_stft():
    def __init__(self):


if __name__ == "__main__":
    #week2DFT().test()
    #week2Sinusoid().test()
    _fft().test()