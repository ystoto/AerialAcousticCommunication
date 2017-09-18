import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'E:/PycharmProjects/sms/software/models'))
#from .stft import stftAnal, stftSynth
import numpy as np
import numpy.random as PN
import datetime
import time
#os.environ['PYTHONASYNCIODEBUG'] = '1'
import asyncio
import wave
import struct

# import importlib.util
# spec = importlib.util.spec_from_file_location("stft", "E:/PycharmProjects/sms/software/models/stft.py")
# STFT = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(STFT)

import utilFunctions as UF
import bitarray
from multiprocessing import Process, Queue
import os
import matplotlib.pyplot as plt

ASCII_MAX = 127
fs = 44100
CHUNK = 4096
frameSize = 1024
pnsize = 32*3  # correlationTest 에 따르면 64 일 때는 118 번째 생성한 PN 에서 오류 발생
partialPnSize = 16*2
maxWatermarkingGain = 0.7  # 0 ~ 1
sync_pn_seed = 8
msg_pn_seed = 9
#SYNC = [+1]
NUMOFSYNCREPEAT = 1
detectionThreshold = 0.6
BASE_FREQ_OF_DATA_EMBEDDING = 17000
FREQ_INTERVAL_OF_DATA_EMBEDDING = 80
NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN = 7  # (7-2) * 23 = 115 msec보다 작거나 같은 주기로 SYNC 모니터링을 해야함.
NUM_OF_FRAMES_PER_PARTIAL_MSG_PN = 3

SYNC = [+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1]

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}


def f(title, data):
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    plt.title(title)
    plt.plot(data)
    plt.show()
    print("plot done")

def plotInNewProcess(title, data, join=False):
    p = Process(target=f, args=(title, data,))
    p.start()
    if join:
        p.join()

def getTargetFreqBand(transformed, targetLength, begin_ratio = 0.7):
    spectrumSize = transformed.size
    if spectrumSize * (1 - begin_ratio) > targetLength:
        begin = int(spectrumSize * begin_ratio)
    else:
        begin = spectrumSize - targetLength
    end = begin + targetLength
    return begin, end

def readWav(inputFile='../../sounds/piano.wav'):
    fs_tmp, x = UF.wavread(inputFile)
    return x


def PRBS(i=4, size=10):
    seed = 39
    PN.seed((71912317 * (i * (i + 2) * (seed + 1)) + (i + 4779) * 317 * (seed)) % 15991)
    return PN.choice([-1, 1], size)

def getPN(i=4, size=10):
    # return PRBS(i,size)
    seed = 39
    PN.seed((71912317 * (i * (i + 2) * (seed + 1)) + (i + 4779) * 317 * (seed)) % 15991)
    result = PN.rand(1, size)
    #print("PN:", result)
    result = result[0]
    #result *= maxWatermarkingGain
    #print("PN:", result)
    return result

def norminnerproduct(a, b):
    num = np.sum(np.multiply(a,b))
    den = np.sum(np.multiply(b,b))
    #num = np.multiply(a, b)
    #den = np.multiply(b, b)
    #print("num:", num)
    return num/den
    #return np.sum(np.divide(num, den))

def SGN(a, b):
    result = norminnerproduct(a, b)
    # print("a:", a[:10])
    # print("b:", b[:10])
    # print("result:", result)
    # # a의 값이 10배가 되면 result도 10배가 됨..
    # if (abs(result) < 0.4):
    #     print("warning!!, bit: %d, accuracy: %f" % (result/abs(result), result))
    if result >= 0:
        return 1, result
    else:
        return -1, result

def printwithsign(msg, ba):
    print (msg, end=" ")
    for i, val in enumerate(ba):
        if val >= 0:
            print("1", end=" ")
        else:
            print("_", end=" ")
    print("")

def printWithIndex(msg, ba, n=False):
    print (msg, end=" ")
    for i, val in enumerate(ba):
        if n:
            v = val
        else:
            v = int(val*100)
        if val >= 0:
            print("%d.(+%02d)" % (i, v), end=" ")
        else:
            print("%d.(-%02d)" % (i, v*-1), end=" ")
    print("")

def print8bitAlignment(ba, printascii = True):
    abyte = []
    for i, val in enumerate(ba):
        if i>0 and i%8 == 0:
            if printascii:
                try:
                    s = bitarray.bitarray(abyte).tostring()
                    print("--> ", s)
                except Exception as err:
                    print(err)
                abyte = []
            else:
                print("")
        print ( "%d" % val, end=" ")
        abyte.append(val)
    print("\n-------")

def convertNegative2Zero(bitssequence):
    for idx, value in enumerate(bitssequence):
        if (bitssequence[idx] == -1):
            bitssequence[idx] = 0
    return bitssequence

def convertZero2Negative(bitssequence):
    for idx, value in enumerate(bitssequence):
        if (bitssequence[idx] == 0):
            bitssequence[idx] = -1
    return bitssequence

def generatePnList():
    if len(generatePnList.pnlist) == 0:
        for i in range(ASCII_MAX):
            generatePnList.pnlist.append(getPN(i+10, pnsize))
        print("pnlist is just generated. ", len(generatePnList.pnlist))
    return np.array(generatePnList.pnlist)
generatePnList.pnlist = []

class RIngBuffer:
    def __init__(self, maxlen = 1024*1024):
        self.maxlen = maxlen
        self.wp = 0
        self.rp = 0
        self.data = np.array([np.float32(None) for i in range(maxlen)])
        self.eos = False

    def dat(self):
        return self.data

    def writeptr(self):
        return self.wp

    def readptr(self):
        return self.rp

    def _WP(self):
        return self.wp % self.maxlen

    def _RP(self):
        return self.rp % self.maxlen

    def _copyIN(self, position, newdata):
        s = len(newdata)
        diff = position + s - self.maxlen
        if diff > 0:
            self.data[position:] = newdata[:-diff]
            self.data[0:diff] = newdata[-diff:]
        else:
            self.data[position:position + s] = newdata

    def write(self, data, allowOverwrite = False, eos = False):
        s = len(data)
        #tprint("write: ", s)
        if self.maxlen - (self.wp - self.rp) >= s: # has enough space
            self._copyIN(self._WP(), data)
            self.wp += s
        else:   # overflow is expected, overwrite and update rp
            self.rp = self.wp + s
            self._copyIN(self._WP(), data)
            self.wp += s
        self.eos = eos

    def read(self, length, update_ptr = True, block = True):
        while True:
            validDataLength = self.wp - self.rp
            if validDataLength >= length:
                break
            elif self.eos or block is False:
                length = validDataLength
                break
            else:
                time.sleep(0.01)

        if length <= 0:
            return np.array([])

        begin = self._RP()
        end = (begin + length) % self.maxlen

        if update_ptr:
            self.rp += length

        if end > begin:
            return self.data[begin:end].copy()
        else:
            return np.append(self.data[begin:], self.data[:end])

    def isAvailablePos(self, position):
        if position >= self.wp:
            return False
        if self.wp - self.maxlen > position:
            return False
        return True

    def readfromSync(self, length, position, update_ptr=True):  # do not update
        while True:
            if self.wp >= position + length:
                break;
            else:
                # print("111", self.wp, self.rp, position)
                time.sleep(0.0005)

        if not self.isAvailablePos(position):
            position = self.wp - self.maxlen
            if position < 0:
                position = 0

        while True:
            validDataLength = self.wp - self.maxlen if self.maxlen < self.wp else self.wp
            if validDataLength < length:
                # print ("222", validDataLength, length, self.wp, self.rp)
                time.sleep(0.0005)
            else:
                break

        begin = position % self.maxlen
        end = (begin + length) % self.maxlen
        # print ("333", self.rp, self.wp, position, length)
        if update_ptr:
            self.rp = position + length

        if end > begin:
            return self.data[begin:end].copy(), position
        else:
            return np.append(self.data[begin:], self.data[:end]), position

    @asyncio.coroutine
    def readfrom(self, length, position, update_ptr = True):  # do not update
        while True:
            if self.wp >= position + length:
                break;
            else:
                #print("111", self.wp, self.rp, position)
                yield from asyncio.sleep(0.0005)

        if not self.isAvailablePos(position):
            position = self.wp-self.maxlen
            if position < 0:
                position = 0

        while True:
            validDataLength = self.wp - self.rp # TODO: Fix as what readfromSync does
            if validDataLength < length:
                #print ("222", validDataLength, length, self.wp, self.rp)
                yield from asyncio.sleep(0.0005)
            else:
                break

        begin = position % self.maxlen
        end = (begin + length) % self.maxlen
        #print ("333", self.rp, self.wp, position, length)
        if update_ptr:
            self.rp = position + length

        if end > begin:
            return self.data[begin:end].copy(), position
        else:
            return np.append(self.data[begin:], self.data[:end]), position

def tprint(*args, **kwargs):
    print(datetime.datetime.now(), end=" ")
    print(*args, **kwargs)

def correlationTest():
    pnarr = generatePnList()

    minCorr = 1
    maxCorr = 0
    sum = 0
    sumOfDiff = 0
    for idx in range(ASCII_MAX):
        max = 0
        secondMax = 0
        maxidx = -1
        a = pnarr[idx]
        corrArray = []
        for zidx in range(ASCII_MAX):
            noise = PN.rand(1, pnsize)[0] / 0.78
            b = np.add(a, noise)  # acoustic noise
            corr = abs(np.corrcoef(b, pnarr[zidx])[0][1])
            if max <= corr:
                max = corr
                maxidx = zidx
            if secondMax < corr < max:
                secondMax = corr

            corrArray.append(corr)

        if minCorr > max:
            minCorr = max
        if maxCorr < max:
            maxCorr = max
        sum += max
        sumOfDiff += (max - secondMax)

        print ("%d  -- idx( %d ), corr( %2.2f , %2.2f ), min( %2.2f ), max( %2.2f ), aver( %2.2f )"\
               % (idx, maxidx, max, secondMax, minCorr, maxCorr, sum/(idx+1)), sumOfDiff/(idx+1))
        # plt.plot(corrArray)
        # plt.show()

        if idx != maxidx:
            print ("***************************************")
            exit(1)

def bitExtractionTestwithShift():
    buf = np.array([np.float32(None) for i in range(pnsize * 3)])
    bits = [-1, +1, -1]
    pn = getPN(2, pnsize)
    for idx in range(3):
        buf[idx * pnsize:(idx + 1) * pnsize] = pn * bits[idx]
        # print (buf[idx*pnsize:(idx+1)*pnsize].round(3))

    for idx in range(9):
        begin = int(idx * (pnsize / 4))
        end = begin + pnsize
        bit, result = SGN(buf[begin:end], pn)
        print(bit, abs(np.corrcoef(buf[begin:end], pn)[0][1]), round(result, 3))

def waveRWtest():
    from freqSpectrum import updateSpectrum
    #wf = wave.open('E:/Dropbox/앱/Hi-Q Recordings/recording-20170703-113452.wav', 'rb')
    wf = wave.open('E:/Dropbox/sin_15khz_1.0.wav', 'rb')
    ch = wf.getnchannels()
    bitwidth = wf.getsampwidth()
    rate = wf.getframerate()
    nframes = wf.getnframes()
    print (ch, bitwidth, rate)
    readframes = 0
    while(True):
        N = 1024
        frames = wf.readframes(N)
        y_ = struct.unpack("%dh" % (len(frames) / bitwidth), frames)
        y = np.array(y_)
        y = np.float32(y) / (2 ** (bitwidth * 8) - 1)
        updateSpectrum(y)
        size = len(y)
        readframes += size
        if readframes >= nframes:
            break
    print("Done")

def generateWave(frequency, amplitude, numOfSamples = 441, samplerate = 44100):
    if len(frequency) != len(amplitude):
        raise Exception("length of parameters Mismatch %d, %d" % (len(frequency), len(amplitude)))
    if amplitude.max() > 1.0 or amplitude.min() < 0.0:
        raise Exception("wrong values in amplitude max: %d, min: %d" % (amplitude.max(), amplitude.min()))

    time = numOfSamples * 1.0 / samplerate
    result = np.zeros(numOfSamples, dtype=np.float64)
    for idx, val in enumerate(frequency):
        w = 2 * np.pi * val
        samples = np.linspace(0, time, numOfSamples)
        result += amplitude[idx] * np.sin(w * samples)

    # normalize
    maxAmp = max(abs(result.max()), abs(result.min()))
    result /= maxAmp
    result *= maxWatermarkingGain
    return result


if __name__ == "__main__":
    #waveRWtest()
    correlationTest()
    #bitExtractionTestwithShift()
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import get_window
    ##############

    from pylab import *
    from numpy.fft import fft, fftshift, ifft, ifftshift

    #
    # emptySpectrum = np.array([np.complex128(0.0) for i in range(int(1024 / 2 + 1))])
    # dataSignal = np.fft.irfft(emptySpectrum)
    # abs(dataSignal.max())

    Fs = 44100.  # the sampling frequency
    Ts = 1. / Fs  # the sampling period

    N = 256  # 샘플 갯수
    freqStep = Fs / N  # resolution of the frequency in frequency domain

    f = 10 * freqStep  # frequency of the wave
    t = arange(N) * Ts  # x ticks in time domain, t = n*Ts
    #y = cos(2 * pi * f * t) + 0.5 * sin(2 * pi * 3 * f * t)  # 테스트 신호
    y = sin(2 * pi * 3 * f * t)  # 테스트 신호
    w = get_window('hamming', N)
    sw = sum(w)


    #w *= y

    Y = fft(y)  # FFT 분석
    W = fft(w)
    #W = np.multiply(W, Y)
    #W *= Y

    Y = fftshift(Y)  # middles the zero-point's axis
    W = fftshift(W)

    figure(figsize=(8, 8))
    subplots_adjust(hspace=.4)

    # Plot time data
    subplot(4, 1, 1)
    plot(t, y, '.-')
    plot(t, w, '.-')
    grid("on")
    xlabel('Time (seconds)')
    ylabel('Amplitude')
    title('signals')
    axis('tight')

    freq = freqStep * arange(-N / 2, N / 2)  # x ticks in frequency domain

    # Plot spectral magnitude
    subplot(4, 1, 2)
    plot(freq, abs(Y), '.-b')
    plot(freq, abs(W))
    grid("on")
    xlabel('Frequency')
    ylabel('Magnitude (Linear)')

    # Plot phase
    subplot(4, 1, 3)
    plot(freq, angle(Y), '.-b')
    plot(freq, angle(W))
    grid("on")
    xlabel('Frequency')
    ylabel('Phase (Radian)')

    # Plot phase
    subplot(4, 1, 4)
    plot(t, ifft(ifftshift(Y)), '.-')
    plot(t, ifft(ifftshift(W)), '-')
    grid("on")
    xlabel('Time (seconds)')
    ylabel('Amplitude')
    title('signals')
    axis('tight')

    show()

    ##################
    plt.close('all')

    fs = 5e5
    print ("fs: ", fs)
    duration = 1
    npts = int(fs * duration)
    t = np.arange(npts, dtype=float) / fs
    f = 1000
    ref = 0.004
    amp = ref * np.sqrt(2)
    signal = amp * np.sin((f * duration) * np.linspace(0, 2 * np.pi, npts))
    # plt.plot(signal)
    # plt.show()

    rms = np.sqrt(np.mean(signal ** 2))
    print("rms: ", rms)
    dbspl = 94 + 20 * np.log10(rms / ref)

    # Window signal
    win = np.hamming(npts)
    signal = signal * win
    plt.plot(signal)
    plt.show()

    sp = np.fft.fft(signal)
    freq = np.fft.fftfreq(npts, 1.0 / fs)

    # Scale the magnitude of FFT by window energy and factor of 2,
    # because we are using half of FFT.
    sp_mag = np.abs(sp) * 2 / np.sum(win)

    # To obtain RMS values, divide by sqrt(2)
    sp_rms = sp_mag / np.sqrt(2)

    # Shift both vectors to have DC at center
    freq = np.fft.fftshift(freq)
    sp_rms = np.fft.fftshift(sp_rms)

    # Convert to decibel scale
    sp_db = 20 * np.log10(sp_rms / ref) + 94

    plt.semilogx(freq, sp_db)
    plt.xlim((0, fs / 2))
    plt.ylim((-100, 100))
    plt.grid('on')

    # Compare the outputs
    print(dbspl, sp_db.max())