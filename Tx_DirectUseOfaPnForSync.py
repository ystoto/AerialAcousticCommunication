import numpy as np
import numpy.random as PN
import os, sys
import datetime
import time
import threading
import wm_util as UT
from scipy.signal import get_window



sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'E:/PycharmProjects/sms/software/models'))


import utilFunctions as UF

from wm_util import pnsize, frameSize, sync_pn_seed, msg_pn_seed, fs, NUMOFSYNCREPEAT, detectionThreshold,\
    subband, partialPnSize, norm_fact, ASCII_MAX, CHUNK, BASE_FREQ_OF_DATA_EMBEDDING, FREQ_INTERVAL_OF_DATA_EMBEDDING,\
    NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN, NUM_OF_FRAMES_PER_PARTIAL_MSG_PN

class NotEnoughData(Exception):
    def __init__(self, *args, **kwargs):
        print(args)
        print(kwargs)
        self.remain_data = None
        super().__init__()


def mix(frameA, frameB):
    result = np.array([np.float32(0.0) for i in frameA])
    for idx, val in enumerate(frameA):
        u = val + frameB[idx] - val * frameB[idx]
        if u > 1.0:
            u = 1.0
        elif u < -1.0:
            u = -1.0
        result[idx] = u
    return result


def insertBit(inbuf, outbuf, partialPN, numOfTargetFrames):

    leftHalfWindow = get_window('hamming', frameSize)
    rightHalfWindow = leftHalfWindow.copy()
    center = int(frameSize/2)
    leftHalfWindow[center:] = leftHalfWindow[center]
    rightHalfWindow[:center] = rightHalfWindow[center]
    # UT.plotInNewProcess("L window", leftHalfWindow)
    # UT.plotInNewProcess("R window", rightHalfWindow)

    # In Freq domain, Generate 3 Frames of New Signal Y which embed data
    freq = [BASE_FREQ_OF_DATA_EMBEDDING + i * FREQ_INTERVAL_OF_DATA_EMBEDDING for i in range(len(partialPN))]
    amp = partialPN
    dataSignal = UT.generateWave(freq, amp, numOfTargetFrames * frameSize, fs)
    # UT.plotInNewProcess("org", dataSignal)

    # wy = windowing to 3 frames
    dataSignal[:frameSize] *= leftHalfWindow
    dataSignal[-frameSize:] *= rightHalfWindow
    # UT.plotInNewProcess("windowed", dataSignal, True)
    if dataSignal.max() > 1.0:
        raise Exception("overflow dataSignal %d" % dataSignal.max())

    # output = mix (org, wy)
    mixedSignal = np.array([np.float32(0.0) for i in range(frameSize * numOfTargetFrames)])
    for i in range(numOfTargetFrames):
        frame = inbuf.read(frameSize)
        if frame.size < frameSize:
            raise NotEnoughData("insertBit", frame.size, "/", frameSize)
        mixedSignal[frameSize * i : frameSize * (i+1)] = mix(frame, dataSignal[frameSize * i : frameSize * (i+1)])

    outbuf.write(mixedSignal)
    # debugging
    # print ("partialPN : ", partialPN)
    # for i in range(numOfFramesPerPartialPN):
    #     b = i * frameSize
    #     e = b + frameSize
    #     print ("extracted : ", extractPartialPN(newSignal[b:e]))

def insertBit1(inbuf, outbuf, partialPN):
    bandsPer1PN = 3

    fullWindow = get_window('hamming', frameSize)
    leftHalfWindow = fullWindow.copy()
    rightHalfWindow = fullWindow.copy()
    center = int(frameSize/2)
    leftHalfWindow[:center] = leftHalfWindow[center]
    rightHalfWindow[center:] = rightHalfWindow[center]

    # In Freq domain, Generate 3 Frames of New Signal Y which embed data
    emptySpectrum = np.array([np.complex128(0.0) for i in range(int(frameSize * NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN / 2 + 1))])
    beginBand, endBand = UT.getTargetFreqBand(emptySpectrum, len(partialPN) * bandsPer1PN, subband[0])
    for idx, value in enumerate(partialPN):
        b = beginBand + idx * bandsPer1PN * NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN
        e = b + bandsPer1PN * NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN
        emptySpectrum.real[b] = value * (len(emptySpectrum.real) * 0.7 / (endBand - beginBand)) # Magnitude 강화
        #print (len(emptySpectrum.real), " ", idx, ". b: %d" % b, ", val: ", emptySpectrum.real[b:b+3])
        #emptySpectrum.imag[b] = emptySpectrum.real[b]

    # IFFT
    dataSignal = np.fft.irfft(emptySpectrum)
    # UT.plotInNewProcess("freq", abs(emptySpectrum))
    # UT.plotInNewProcess("org", dataSignal)

    # wy = windowing to 3 frames
    dataSignal[:frameSize] *= leftHalfWindow
    dataSignal[-frameSize:] *= rightHalfWindow
    dataSignal /= max(abs(dataSignal.max()), abs(dataSignal.min()))
    # UT.plotInNewProcess("L window", leftHalfWindow)
    # UT.plotInNewProcess("R window", rightHalfWindow)
    # UT.plotInNewProcess("C window", fullWindow)
    # UT.plotInNewProcess("windowed", dataSignal, True)

    # output = mix (org, wy)
    mixedSignal = np.array([np.float32(0.0) for i in range(frameSize * NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN)])
    for i in range(NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN):
        frame = inbuf.read(frameSize)
        if frame.size < frameSize:
            raise NotEnoughData("insertBit", frame.size, "/", frameSize)
        mixedSignal[frameSize * i : frameSize * (i+1)] = mix(frame, dataSignal[frameSize * i : frameSize * (i+1)])

    outbuf.write(mixedSignal)
    # debugging
    # print ("partialPN : ", partialPN)
    # for i in range(numOfFramesPerPartialPN):
    #     b = i * frameSize
    #     e = b + frameSize
    #     print ("extracted : ", extractPartialPN(newSignal[b:e]))

def insertBit0(inbuf, outbuf, partialPN):
    bandsPer1PN = 3
    hopSize = 512

    orgSignal = np.array([np.float32(0.0) for i in range(frameSize * NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN)])
    newSignal = np.array([np.float32(0.0) for i in range(frameSize * NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN)])
    begin = 0
    filledOrgSignal = 0

    fullWindow = get_window('hamming', frameSize)
    leftHalfWindow = fullWindow.copy()
    rightHalfWindow = fullWindow.copy()
    center = int(frameSize/2)
    leftHalfWindow[:center] = leftHalfWindow[center]
    rightHalfWindow[center:] = rightHalfWindow[center]

    # normalized window
    fullWindow = fullWindow / sum(fullWindow)
    leftHalfWindow = leftHalfWindow / sum(leftHalfWindow)
    rightHalfWindow = rightHalfWindow / sum(rightHalfWindow)
    bo = True
    while (begin + frameSize <= frameSize * NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN): # N frames 에 hop size 만큼 이동하면서 partialPN을 입력
        if filledOrgSignal < begin + frameSize:
            frame = inbuf.read(frameSize);
            if frame.size < frameSize:
                raise NotEnoughData("insertBit", frame.size, "/", frameSize)
            orgSignal[ filledOrgSignal:filledOrgSignal+frameSize ] = frame
            filledOrgSignal += frameSize

        AllowedFrameToEmbedData = True
        # Windowing
        end = begin + frameSize
        if begin == 0: # left half는 윈도우 제거
            windowedSignal = orgSignal[begin:end] * leftHalfWindow
            AllowedFrameToEmbedData = False
        elif begin + frameSize == frameSize * NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN:  # right half는 윈도우 제거
            windowedSignal = orgSignal[begin:end] * rightHalfWindow
            AllowedFrameToEmbedData = False
        else:
            windowedSignal = orgSignal[begin:end] * fullWindow
            AllowedFrameToEmbedData = True


        # Time to Frequency domain transform
        targetSpectrum = np.fft.rfft(windowedSignal)
        # if bo:
        #     UT.plotInNewProcess("org", orgSignal[begin:end])
        #     UT.plotInNewProcess("win", windowedSignal)
        #     UT.plotInNewProcess("tSpec", targetSpectrum.real)

        # Data Embedding
        for band in subband: # embed a bit to multiple subbands
            beginBand, endBand = UT.getTargetFreqBand(targetSpectrum, len(partialPN) * bandsPer1PN, band)
            for idx, value in enumerate(partialPN):
                b = beginBand + idx * bandsPer1PN
                e = b + bandsPer1PN
                if AllowedFrameToEmbedData is True:
                    #targetSpectrum.real[b:e] = value * (len(targetSpectrum.real) * 0.7 / (endBand - beginBand)) # Magnitude 강화
                    targetSpectrum.real[256] = 128.0

        # Frequency to Time domain transform
        newSignal[begin:end] += np.fft.irfft(targetSpectrum)# * hopSize
        #print ("max : ", newSignal[begin:end], newSignal.max())

        # if bo:
        #     bo = False
        #     UT.plotInNewProcess("tSpec2", targetSpectrum.real)
        #     UT.plotInNewProcess("partialPN", partialPN)
        #     UT.plotInNewProcess("new", newSignal[begin:end], True)
        begin += hopSize

    outbuf.write(newSignal)
    # debugging
    # print ("partialPN : ", partialPN)
    # for i in range(numOfFramesPerPartialPN):
    #     b = i * frameSize
    #     e = b + frameSize
    #     print ("extracted : ", extractPartialPN(newSignal[b:e]))


def extractPartialPN(signal):
    Y = np.fft.rfft(signal)
    bandsPer1PN = 3
    found = []
    for i in range(partialPnSize):
        freq = BASE_FREQ_OF_DATA_EMBEDDING + i * FREQ_INTERVAL_OF_DATA_EMBEDDING
        bandDuration = fs / 2 / len(Y)
        found.append(abs(Y[int(freq / bandDuration)]))
    return found

def insertSYNC(inbuf, outbuf):
    pn = UT.getPN(sync_pn_seed, pnsize)
    for repeat in range(NUMOFSYNCREPEAT):
        for idx in range(int(pnsize / partialPnSize)):
            begin = idx * partialPnSize
            end = begin + partialPnSize
            insertBit(inbuf, outbuf, pn[begin:end], NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN)

def insertMSG(inbuf, outbuf, msg):
    pnlist = UT.generatePnList()
    nextPosition = 0
    print("****", end=" ")
    for idx, value in enumerate(msg):
        asciiCode = ord(value)
        print("%c[ %d ]" % (value, asciiCode), end=" ")
        pn = pnlist[asciiCode]
        for idx in range(int(pnsize / partialPnSize)):
            begin = idx * partialPnSize
            end = begin + partialPnSize
            insertBit(inbuf, outbuf, pn[begin:end], NUM_OF_FRAMES_PER_PARTIAL_MSG_PN)
    print("")
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

def extractPN(outbuf, position):
    found = []
    for repeat in range(NUMOFSYNCREPEAT):
        for idx in range(int(pnsize / partialPnSize)):
            singal, newPos = outbuf.readfromSync(frameSize, position, update_ptr=False)
            result = extractPartialPN(singal)
            found.extend(result)
            position += frameSize * NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN
    return found

def insertWM(inbuf, outbuf, msg=" "):
    try:
        out_writeptr = outbuf.writeptr()
        print("Insert SYNC - begin", inbuf.readptr(), outbuf.writeptr())
        insertSYNC(inbuf, outbuf)
        print("Insert SYNC - end", inbuf.readptr(), outbuf.writeptr())

        pn = UT.getPN(sync_pn_seed, pnsize)
        print("pn: ", pn)
        interval = int(frameSize / 8)
        for di in range(int(frameSize / interval) * 2 + 1):
            ePN = extractPN(outbuf, out_writeptr + interval * di)
            print("Pos: ", out_writeptr + interval * di, "ePN%d : " % di, abs(np.corrcoef(pn, ePN)[0][1]), ", epn:", ePN);

        out_writeptr = outbuf.writeptr()
        print("Insert MSG - begin", inbuf.readptr(), outbuf.writeptr())
        insertMSG(inbuf, outbuf, msg)
        print("Insert MSG - end", inbuf.readptr(), outbuf.writeptr())
        # extractMSG(outbuf, out_writeptr)
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
                data = self.inbuf.read(frameSize)
                #print("wm - read - end - ", data.size)
                transfered_size += data.size
                if data.size == 0:
                    print("wm - done")
                    break
                # print("wm - write - begin - ", data.size)
                self.outbuf.write(data)
                #print("wm - write - end")

def Start(inbuf, outbuf):
    wm = watermaker(inbuf, outbuf)
    wm.start()
    return wm


if __name__ == "__main__":
    inputFile = 'SleepAway_partial.wav'
    #inputFile = 'silence.wav'
    input_signal = UT.readWav(inputFile)

    src = UT.RIngBuffer(fs * 60)
    sink = UT.RIngBuffer(fs * 600)

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
        src.write(input_signal[:write_size], eos=False if input_signal.size > CHUNK else True)
        transfered_size += write_size
        input_signal = input_signal[write_size:]
        time.sleep(0.01)

    thread.join()
    print("thread terminated")

    # watermarker thread가 처리 완료한 데이타를 SINK ringbuffer로부터 읽어들임.
    # print("A - READ")
    while(True):
        a = sink.read(100 * CHUNK, block=False)
        if a.size == 0:
            break
        watermarked_data.extend(a)

    print("all data is written")
    # output sound file (monophonic with sampling rate of 44100)
    outputFile = './' + os.path.basename(inputFile)[:-4] + '_stft.wav'
    UF.wavwrite(np.array(watermarked_data), fs, outputFile)