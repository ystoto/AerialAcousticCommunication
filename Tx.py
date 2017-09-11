# function to call the main analysis/synthesis functions in software/models/stft.py
import numpy as np
import numpy.random as PN
import os, sys
import bitarray
import datetime
import time
import threading
import wm_util as UT
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'E:/PycharmProjects/sms/software/models'))
#from .stft import stftAnal, stftSynth

# import importlib.util
# spec = importlib.util.spec_from_file_location("stft", "E:/PycharmProjects/sms/software/models/stft.py")
# STFT = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(STFT)

import utilFunctions as UF

from wm_util import pnsize, frameSize, sync_pn_seed, msg_pn_seed, fs, SYNC, NUMOFSYNCREPEAT, detectionThreshold,\
    subband, partialPnSize, norm_fact, ASCII_MAX, CHUNK

class NotEnoughData(Exception):
    def __init__(self, *args, **kwargs):
        print(args)
        print(kwargs)
        self.remain_data = None
        super().__init__()

def insertBit(inbuf, outbuf, bit, pn, overwrite = False):
    numOfPartialPNs = int((pnsize + partialPnSize - 1) / partialPnSize)
    embededData = pn * bit

    for idx in range(numOfPartialPNs):
        begin = idx * partialPnSize
        end = begin + partialPnSize
        if end > pnsize:
            end = pnsize
        embededDataLength = end - begin
        partialEmbededData = embededData[begin:end]

        targetFrame = inbuf.read(frameSize)
        if targetFrame.size < frameSize:
            raise NotEnoughData("insertBit", outbuf, targetFrame)
        targetSpectrum = np.fft.fft(targetFrame)

        for band in subband: # embed a bit to multiple subbands
            begin, end = UT.getTargetFreqBand(targetSpectrum, embededDataLength, band)
            if overwrite == True:
                targetSpectrum.real[begin:end] = partialEmbededData * (len(targetSpectrum) * 0.7 / (end - begin)) # Magnitude 강화
            else:
                targetSpectrum.real[begin:end] += partialEmbededData

        u = np.fft.ifft(targetSpectrum)
        outbuf.write(np.fft.ifft(targetSpectrum).real)
        # plt.plot(ts[256:, 1]+0.001)
        # plt.show()


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

def insertSYNC(inbuf, outbuf):
    for repeat in range(NUMOFSYNCREPEAT):
        for idx, value in enumerate(SYNC):
            #print("before:::", target[idx * framelength:idx * framelength+10])
            pn = UT.getPN(sync_pn_seed, pnsize)
            insertBit(inbuf, outbuf, value, pn, overwrite=True)
            #print("after:::", target[idx * framelength:idx * framelength + 10])


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

def extractSYNC(outbuf, position):
    found = []
    for bitIdx, value in enumerate(SYNC):
        pn = UT.getPN(sync_pn_seed, pnsize)
        numOfPartialPNs = int((pnsize + partialPnSize - 1) / partialPnSize)
        embededData = []

        for idx in range(numOfPartialPNs):
            begin = idx * partialPnSize
            end = begin + partialPnSize
            if end > pnsize:
                end = pnsize
            embededDataLength = end - begin

            srcBegin = position + idx * frameSize
            srcEnd = srcBegin + frameSize
            targetFrame, realPos = outbuf.readfromSync(frameSize, srcBegin, update_ptr=False)
            srcSpectrum = np.fft.fft(targetFrame)

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

def insertMSG(inbuf, outbuf, msg):
    pnlist = UT.generatePnList()
    nextPosition = 0
    print("****", end=" ")
    for idx, value in enumerate(msg):
        asciiCode = ord(value)
        print("%c[ %d ]" % (value, asciiCode), end=" ")
        pn = pnlist[asciiCode]
        nextPosition = insertBit(inbuf, outbuf, 1, pn, overwrite=True)
    print("")
    return nextPosition

def extractMSG(outbuf, position):
    idx = 0
    numOfPartialPNs = int((pnsize + partialPnSize - 1) / partialPnSize)
    pnlist = UT.generatePnList()
    result = str("")
    while (True):
        begin = position + (idx * frameSize * numOfPartialPNs)
        end = begin + frameSize * numOfPartialPNs
        idx += 1

        embededCode = [[] for i in subband]
        for frameIdx in range(numOfPartialPNs):
            b = begin + frameIdx * frameSize
            e = b + frameSize
            frame, realPos = outbuf.readfromSync(frameSize, b, update_ptr=False)
            transformed = np.fft.fft(frame)
            for num, band in enumerate(subband):
                b, e = UT.getTargetFreqBand(transformed, partialPnSize, band)
                embededCode[num].extend(transformed.real[b:e])

        maxCorr = 0
        maxCorrIdx = -1
        for num, band in enumerate(subband):
            for i, value in enumerate(pnlist):
                corr = abs(np.corrcoef(embededCode[num], value)[0][1])
                if corr > maxCorr:
                    maxCorr = corr
                    maxCorrIdx = i

        character = chr(maxCorrIdx)
        print("%c[ %d ]" % (character, maxCorrIdx), end=" ")
        if character == '\n':
            break
        result += character
    print("")
    return result, end

def insertWM(inbuf, outbuf, msg=" "):
    try:
        out_writeptr = outbuf.writeptr()
        print("Insert SYNC - begin", inbuf.readptr(), outbuf.writeptr())
        insertSYNC(inbuf, outbuf)
        print("Insert SYNC - end", inbuf.readptr(), outbuf.writeptr())
        extractSYNC(outbuf, out_writeptr)

        out_writeptr = outbuf.writeptr()
        print("Insert MSG - begin", inbuf.readptr(), outbuf.writeptr())
        insertMSG(inbuf, outbuf, msg)
        print("Insert MSG - end", inbuf.readptr(), outbuf.writeptr())
        extractMSG(outbuf, out_writeptr)
    except Exception as err:
        print(err)


class watermaker(threading.Thread):
    def __init__(self, inbuf, outbuf):
        self.inbuf = inbuf
        self.outbuf = outbuf
        self.wm_requested = False
        self.requested_msg = ""
        super().__init__()

    def requestWM(self, msg):  # msg must include '\n'
        self.requested_msg = msg
        self.wm_requested = True

    def run(self):
        print ("watermaker is running")
        transfered_size = 0
        while(True):
            if self.wm_requested:
                print("Insert watermark - begin", self.inbuf.readptr(), self.outbuf.writeptr())
                insertWM(self.inbuf, self.outbuf, self.requested_msg)
                print("Insert watermark - end", self.inbuf.readptr(), self.outbuf.writeptr())
                self.wm_requested = False
                self.requested_msg = ""
            else:
                # watermark 요청이 없으면 inbuf 에서 outbuf로 data bypass.
                #print("wm - read - begin")
                data = self.inbuf.read(CHUNK)
                #print("wm - read - end - ", data.size)
                transfered_size += data.size
                if data.size == 0:
                    print("wm - done")
                    break
                #print("wm - write - begin - ", data.size)
                self.outbuf.write(data)
                #print("wm - write - end")

def Start(inbuf, outbuf):
    wm = watermaker(inbuf, outbuf)
    wm.start()
    return wm


if __name__ == "__main__":
    #inputFile = 'SleepAway_partial.wav'
    inputFile = 'silence.wav'
    input_signal = UT.readWav(inputFile)

    src = UT.RIngBuffer(44100 * 60)
    sink = UT.RIngBuffer(44100 * 60)

    wm_position = 3000 # msg 삽입할 위치 (단위 : msec)
    wm_position = int((fs / 100) * (wm_position / 10))  # 단위변환 / in 10 msec

    # insert WM
    thread = Start(src, sink)  # TODO: 1.2 second
    print("Thread is requested")
    transfered_size = 0
    total_size = input_signal.size
    wm_requested = False
    watermarked_data = []

    while (input_signal.size > 0):
        if wm_requested == False and wm_position <= transfered_size:
            wm_requested = True
            thread.requestWM("www.naver.com\n")

        # chunk 단위로 wav를 SRC ringbuffer에 입력 (watermarker에 의해 처리될 수 있도록)
        #print ("A - WRITE done( %d )remained( %d ) total( %d ), " % (transfered_size, input_signal.size, total_size))
        write_size = CHUNK if input_signal.size >= CHUNK else input_signal.size
        src.write(input_signal[:write_size], eos = False if input_signal.size > CHUNK else True)
        transfered_size += write_size
        input_signal = input_signal[write_size:]

        # watermarker thread가 처리 완료한 데이타를 SINK ringbuffer로부터 읽어들임.
        #print("A - READ")
        watermarked_data.extend(sink.read(write_size))

    thread.join()
    print("thread terminated")

    print("all data is written")
    # output sound file (monophonic with sampling rate of 44100)
    outputFile = './' + os.path.basename(inputFile)[:-4] + '_stft.wav'
    UF.wavwrite(np.array(watermarked_data), fs, outputFile)