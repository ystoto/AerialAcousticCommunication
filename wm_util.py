import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'E:/PycharmProjects/sms/software/models'))
#from .stft import stftAnal, stftSynth
import numpy as np
import numpy.random as PN
import datetime
#os.environ['PYTHONASYNCIODEBUG'] = '1'
import asyncio

import importlib.util
spec = importlib.util.spec_from_file_location("stft", "E:/PycharmProjects/sms/software/models/stft.py")
STFT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STFT)

import utilFunctions as UF
import bitarray

fs = 44100
frameSize = 1024
pnsize = 128  # 64bit 이상으로 올리면 스펙트럼 왜곡 발생함.., 대신 dB 을 0.2 정도로 낮추면 32bit 수준으로 왜곡 줄어듬.
partialPnSizePerFrame = 32
maxWatermarkingGain = 2  # 0 ~ 1
sync_pn_seed = 1
msg_pn_seed = 2
#SYNC = [+1]
NUMOFSYNCREPEAT = 2
SYNC = [+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1]
# SYNC = [-1,+1,+1,+1,-1,+1,-1,+1,-1,-1,
#         +1,-1,-1,+1,+1,+1,-1,+1,-1,+1,
#         -1,-1,+1,-1,+1,-1,-1,+1,+1,-1,
#         +1,-1,-1,+1,+1,-1,+1,+1,+1,+1,
#         +1,+1,-1,+1,+1,+1,-1,+1,-1,+1,
#         +1,-1,-1,-1,-1,-1,-1,-1,+1,+1,
#         +1,-1,-1,-1]
#SYNC = [1 for i in range(32)]
detectionThreshold = 0.7
subband = [0.6]

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

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
    #print ("PN:", result)
    result = result[0]
    #print ("PN:", result)
    return result * maxWatermarkingGain

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



class RIngBuffer:
    def __init__(self, maxlen = 1024*1024):
        self.maxlen = maxlen
        self.wp = 0
        self.rp = 0
        self.data = np.array([np.float32(None) for i in range(maxlen)])

    def dat(self):
        return self.data

    def writeptr(self):
        return self.wp

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

    def write(self, data, allowOverwrite = False):
        s = len(data)
        #tprint("write: ", s)
        if self.maxlen - (self.wp - self.rp) >= s: # has enough space
            self._copyIN(self._WP(), data)
            self.wp += s
        else:   # overflow is expected, overwrite and update rp
            self.rp = self.wp + s
            self._copyIN(self._WP(), data)
            self.wp += s

    async def read(self, length):
        while True:
            validDataLength = self.wp - self.rp
            if validDataLength < length:
                await asyncio.sleep(0.01)
            else:
                break

        begin = self._RP()
        end = (begin + length) % self.maxlen

        self.rp += length

        if end > begin:
            return self.data[begin:end].copy()
        else:
            return self.data[begin:].copy().append(self.data[:end])

    def isAvailablePos(self, position):
        if position >= self.wp:
            return False
        if self.wp - self.maxlen > position:
            return False
        return True

    async def read(self, length, position):  # do not update
        while True:
            if self.wp >= position + length:
                break;
            else:
                #print("111", self.wp, self.rp, position)
                await asyncio.sleep(0.0005)

        if not self.isAvailablePos(position):
            position = self.wp-self.maxlen

        while True:
            validDataLength = self.wp - self.rp
            if validDataLength < length:
                #print ("222", validDataLength, length, self.wp, self.rp)
                await asyncio.sleep(0.0005)
            else:
                break

        begin = position % self.maxlen
        end = (begin + length) % self.maxlen
        #print ("333", self.rp, self.wp, position, length)
        self.rp = position + length

        if end > begin:
            return self.data[begin:end].copy(), position
        else:
            return self.data[begin:].copy().append(self.data[:end]), position

def tprint(*args, **kwargs):
    print(datetime.datetime.now(), end=" ")
    print(*args, **kwargs)

def correlationTest():
    localpnsize = 64
    pnarr = []
    for idx in range(127):
        pn = getPN(idx, localpnsize)
        pnarr.append(pn)

    for idx in range(127):
        max = 0
        maxidx = -1
        a = pnarr[idx]
        for zidx in range(127):
            noise = PN.rand(1, localpnsize)[0] / 0.78
            b = np.add(a, noise)  # acoustic noise
            corr = abs(np.corrcoef(b, pnarr[zidx])[0][1])
            if max <= corr:
                max = corr
                maxidx = zidx
        print ("%d  -- %d" % (idx, maxidx), max)
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

if __name__ == "__main__":
    #correlationTest()
    bitExtractionTestwithShift()
