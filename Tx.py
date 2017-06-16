# function to call the main analysis/synthesis functions in software/models/stft.py
import numpy as np
import numpy.random as PN
import os, sys
import bitarray
import datetime
import wm_util as UT
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'E:/PycharmProjects/sms/software/models'))
#from .stft import stftAnal, stftSynth

import importlib.util
spec = importlib.util.spec_from_file_location("stft", "E:/PycharmProjects/sms/software/models/stft.py")
STFT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STFT)

import utilFunctions as UF
import mdct as MDCT

from wm_util import pnsize, frameSize, sync_pn_seed, msg_pn_seed, fs, SYNC, NUMOFSYNCREPEAT, detectionThreshold,\
    subband, partialPnSizePerFrame, norm_fact

def insertBit(sourceSignal, position, bit, pnSeed, overwrite = False):
    pn = UT.getPN(pnSeed, pnsize)
    numOfPartialPNs = int((pnsize + partialPnSizePerFrame - 1) / partialPnSizePerFrame)
    embededData = pn * bit

    for idx in range(numOfPartialPNs):
        begin = idx * partialPnSizePerFrame
        end = begin + partialPnSizePerFrame
        if end > pnsize:
            end = pnsize
        embededDataLength = end - begin
        partialEmbededData = embededData[begin:end]

        srcBegin = position + idx * frameSize
        srcEnd = srcBegin + frameSize
        targetSpectrum = np.fft.fft(sourceSignal[srcBegin:srcEnd])

        for band in subband: # embed a bit to multiple subbands
            begin, end = UT.getTargetFreqBand(targetSpectrum, embededDataLength, band)
            if overwrite == True:
                targetSpectrum.real[begin:end] = partialEmbededData
            else:
                targetSpectrum.real[begin:end] += partialEmbededData

        u = np.fft.ifft(targetSpectrum)
        sourceSignal[srcBegin:srcEnd] = np.fft.ifft(targetSpectrum).real
        # plt.plot(ts[256:, 1]+0.001)
        # plt.show()

    return srcEnd

def insertBitOld(sourceSignal, bit, pnSeed, framesize, overwrite = False):
    pn = UT.getPN(pnSeed, pnsize)
    #print("insertBit, pn:", pn[:5])

    if (True): # Basic Algorithm
        #watermarkedSignal = np.add(originalSignal, np.multiply(pn, bit))
        sourceSpectrum = np.fft.fft(sourceSignal)

        #plt.plot(sourceSpectrum)
        adjustedSpectrum = sourceSpectrum.copy()
        # plt.plot(adjustedSpectrum[:,1])

        for band in subband:
            begin, end = UT.getTargetFreqBand(adjustedSpectrum, pnsize, band)
            if overwrite == True:
                adjustedSpectrum.real[begin:end] = pn * bit
            else:
                adjustedSpectrum.real[begin:end] += (pn * bit / 10000)
        # plt.plot(adjustedSpectrum[:, 1])

        #print ("%d -> %d " % (bit, UT.SGN(adjustedSpectrum[begin:end, 1], pn)))

        watermarkedSignal = np.fft.ifft(adjustedSpectrum).real

        # l1, = plt.plot(sourceSignal, label="L1")
        # l2, = plt.plot(watermarkedSignal, label='l2')
        # l3, = plt.plot(watermarkedSignal - sourceSignal, label='diff')
        # plt.legend(handles = [l1,l2,l3])
        # plt.show()

        #watermarkedSignal = np.multiply(sourceSignal, np.multiply(pn, bit))
        #watermarkedSignal = np.multiply(pn, bit)
        #watermarkedSignal = pn
    else:   # improved algorithm (Not work well)
        a1 = 0.005
        a2 = 0.001
        nlambda = 0.9
        psi = UT.norminnerproduct(sourceSignal, pn)
        if (psi >= 0 and bit == 1):
            watermarkedSignal = np.add(sourceSignal, np.multiply(pn, a1))
        elif (psi >= 0 and bit == -1):
            watermarkedSignal = np.add(sourceSignal, np.multiply(pn, -1 * (a2 + nlambda * psi)))
        elif (psi < 0 and bit == -1):
            watermarkedSignal = np.add(sourceSignal, np.multiply(pn, -1 * a1))
        elif (psi < 0 and bit == 1):
            watermarkedSignal = np.add(sourceSignal, np.multiply(pn, -1 * (a2 - nlambda * psi)))

    return watermarkedSignal

def insertSYNC(target):
    nextPosition = 0
    for repeat in range(NUMOFSYNCREPEAT):
        for idx, value  in enumerate(SYNC):
            #print("before:::", target[idx * framelength:idx * framelength+10])
            nextPosition = insertBit(target, nextPosition, value, sync_pn_seed, overwrite=True)
            #print("after:::", target[idx * framelength:idx * framelength + 10])

    return nextPosition


def findSYNC(source):
    found = []
    max_value = 0
    max_index = -1
    pn = UT.getPN(sync_pn_seed, pnsize) * 100
    #print("pn: ", pn[:20])
    corrarray = []
    aa  = datetime.datetime.now()
    for idx in range(frameSize + 2):
        transformed = np.fft.fft(source[idx:idx + int(frameSize)])
        begin, end = UT.getTargetFreqBand(transformed, pnsize, subband[0])
        corr = abs(np.corrcoef(transformed.real[begin:end], pn)[0][1])
        #corr = np.correlate(transformed[begin:end, 1], pn)
        corrarray.append(corr)
        if max_value <= corr:
            max_value = corr
            max_index = idx
    bb = datetime.datetime.now()
    print("execution time: ", bb-aa)
    print ("max: ", max_value, max_index)
    print("pn_max/min : ", pn.max(), pn.min())
    # plt.plot(corrarray)
    # plt.show()

    if max_value > detectionThreshold:
        return max_index
    else:
        print("Can't find cross-correlation peak over 0.8, max:%d in idx:%d" % (max_value, max_index))
        return -1

def addNoise(source):
    print("Add Noise\ninput:", source[:10])
    noise = PN.rand(1, frameSize)[0] / 10
    frame = np.add(source[0:frameSize], noise)  # acoustic noise
    print("noise:", noise[:10])
    print("output:", frame[:10])

def extractSYNC(sourceSignal):
    found = []
    position = 0
    for bitIdx, value in enumerate(SYNC):
        pn = UT.getPN(sync_pn_seed, pnsize)
        numOfPartialPNs = int((pnsize + partialPnSizePerFrame - 1) / partialPnSizePerFrame)
        embededData = []

        for idx in range(numOfPartialPNs):
            begin = idx * partialPnSizePerFrame
            end = begin + partialPnSizePerFrame
            if end > pnsize:
                end = pnsize
            embededDataLength = end - begin

            srcBegin = position + idx * frameSize
            srcEnd = srcBegin + frameSize
            srcSpectrum = np.fft.fft(sourceSignal[srcBegin:srcEnd])

            begin, end = UT.getTargetFreqBand(srcSpectrum, embededDataLength, subband[0])
            embededData.extend(srcSpectrum.real[begin:end])
        position = srcEnd

        result, accuracy = UT.SGN(embededData, pn)
        if (result == value):
            found.append(result)
        else:
            break

    UT.printwithsign("SYNC:", SYNC)
    UT.printwithsign("BITS:", found)
    if (found == SYNC):
        print("SYNC is found")
        return True
    else:
        print("SYNC is not found: ", found)
        return False

def insertDATA(target, msg):
    ba = bitarray.bitarray()
    ba.fromstring(msg)
    bitlist = ba.tolist()
    UT.convertZero2Negative(bitlist)
    #UT.print8bitAlignment(bitlist)
    wptr = 0
    for idx, value in enumerate(bitlist):
        target[idx * frameSize : (idx + 1) * frameSize] =\
            insertBitOld(target[idx * frameSize : (idx + 1) * frameSize], value, msg_pn_seed, frameSize, overwrite=True)
        #nextPosition = insertBit(target, nextPosition, value, sync_pn_seed, overwrite=True)
        wptr = (idx + 1) * frameSize
    return wptr


def insertWM(rawdata, location_msec = 5000, msg=" "): #location in msec
    sample_location = int((fs / 100) * (location_msec / 10))  # in 10 msec
    target = rawdata[sample_location:]
    print("Pos: %s\nBefore WM:" % (sample_location), rawdata[sample_location:sample_location + 60].round(3))

    wptr = insertSYNC(target)
    print("After WM :", rawdata[sample_location:sample_location + 60].round(3))

    #addNoise(rawdata[220500:])
    extractSYNC(target)
    #findSYNC(rawdata[220500-512:])

    target = target[wptr:] # update write pointer

    insertDATA(target, msg)

    # found = []
    # for idx, value in enumerate(range(120)):
    #     frame = target[idx * frameSize : (idx + 1) * frameSize]
    #     pn = UT.getPN(data_pn_seed, pnsize)
    #     transformed = MDCT.mdct(frame)
    #     begin, end = UT.getTargetFreqBand(transformed)
    #     result = UT.SGN(transformed[begin:end,1], pn)
    #     found.append(result)
    # UT.convertNegative2Zero(found)
    # UT.print8bitAlignment(found)
    # Rx.findMSG(target, 0)

if __name__ == "__main__":
    inputFile = 'SleepAway_partial.wav'
    input_signal = UT.readWav(inputFile)
    print (type(input_signal))
    insertWM(input_signal, 5000, "www.naver.com\n")

    # output sound file (monophonic with sampling rate of 44100)
    outputFile = './' + os.path.basename(inputFile)[:-4] + '_stft.wav'
    UF.wavwrite(input_signal, fs, outputFile)