import struct
import wave
import numpy as np

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.signal import hann
import numpy as np
import time

SAVE = 0.0
TITLE = ''
FPS = 25.0

nFFT = 512
BUF_SIZE = 4 * nFFT
CHANNELS = 1
RATE = 44100
initSpectrum = True
totalFrame = None

def updateSpectrum(orgframe, minsize = 0):
    global line
    global fig
    global initSpectrum
    global totalFrame
    if initSpectrum:
        print("init")
        #plt.ion()
        fig = plt.figure()
        x_f = np.linspace(0, RATE / 2.0, nFFT/2)
        ax = fig.add_subplot(111, title="spectrum", xlim=(x_f[0], x_f[-1]),
                             ylim=(0, 2 * np.pi * nFFT ** 2 / RATE))
        #ax.set_yscale('symlog', linthreshy=nFFT ** 0.5)
        line, = ax.plot(x_f, np.zeros(nFFT/2))
        plt.xlabel("Frequency(Hz)")
        plt.ylabel("Power(dB)")
        line.set_ydata(np.zeros(nFFT))
        initSpectrum = False

    frame = orgframe
    resetTotalFrame = False
    if minsize > 0:
        if totalFrame == None:
            totalFrame = []
        totalFrame.extend(frame)
        if len(totalFrame) < minsize:
            return
        else:
            frame = np.array(totalFrame)
            resetTotalFrame = True

    #Y = np.abs(np.fft.fft(frame, nFFT))
    Y = np.abs(np.fft.rfft(frame, nFFT))
    Y = 20 * np.log10(Y)
    line.set_ydata(Y[:-1])
    plt.pause(0.000001)

    if resetTotalFrame:
        totalFrame = None

def generateSamples(freq):
    Hz = freq
    return (np.sin(2 * np.pi * np.arange(RATE * 5.0) * Hz / RATE)).astype(np.float32)
    #return np.random.rand(1, 1024)[0]

def showBoundaryOfFFT():
    samples = generateSamples(2205)
    left = samples[:1024]
    center = samples[1024:1024 * 2]
    right = samples[1024 * 2:1024 * 3]
    ax1 = plt.subplot(211)
    ax1.plot(np.append(np.append(left, center), right))

    w = np.hanning(1024)
    X = np.fft.rfft(center*w, 1024)
    X[256:258] = 80.0
    center = np.fft.irfft(X)
    ax2 = plt.subplot(212)
    ax2.plot(np.append(np.append(left, center), right))
    plt.show()


if __name__ == '__main__':
    showBoundaryOfFFT()
    # for i in range(88):
    #     samples = generateSamples(440 + i * 100)
    #     updateSpectrum(samples, 44100)
    # plt.show()