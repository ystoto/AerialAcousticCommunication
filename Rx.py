# function to call the main analysis/synthesis functions in software/models/stft.py
import numpy as np
import os, sys
import bitarray
import datetime
import time
import subprocess
import pyaudio
import matplotlib.pyplot as plt

#os.environ['PYTHONASYNCIODEBUG'] = '1'
import asyncio

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'E:/PycharmProjects/sms/software/models'))
#from .stft import stftAnal, stftSynth

# import importlib.util
# spec = importlib.util.spec_from_file_location("stft", "E:/PycharmProjects/sms/software/models/stft.py")
# STFT = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(STFT)

import utilFunctions as UF
import freqSpectrum as FS
import wm_util as UT
from wm_util import pnsize, frameSize, sync_pn_seed, msg_pn_seed, fs, NUMOFSYNCREPEAT, detectionThreshold,\
    norm_fact, partialPnSize, CHUNK, NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN, NUM_OF_FRAMES_PER_PARTIAL_MSG_PN,\
    BASE_FREQ_OF_DATA_EMBEDDING, FREQ_INTERVAL_OF_DATA_EMBEDDING

#inputFile = 'last_mic_in5.wav'
#inputFile = 'SleepAway_partial.wav'
inputFile = 'SleepAway_partial_stft.wav'
#inputFile = 'silence_stft.wav'
#inputFile = 'after.wav'
USE_MIC = True
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
TIMEOUT = 0
watingtime = frameSize * (NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN-2) / RATE # 1024 * (7-2) / 44100 = 116ms Now, the getmaxcorr needs just 0.145msec

@asyncio.coroutine
def printMSG(posRead):
    print ("---------------------------")
    source, realPos = yield from rb.readfrom(frameSize * 88, posRead)
    ix = 0
    ppn = UT.getPN(msg_pn_seed, pnsize)
    bitseq = []
    for n in range(88):
        begin = ix + n*frameSize
        end = begin + frameSize
        fram = source[begin:end]
        tr = np.fft.fft(fram)
        begin, end = UT.getTargetFreqBand(tr, pnsize, subband[0])
        re, accu = UT.SGN(tr.real[begin:end], ppn)
        bitseq.append(re)
    UT.convertNegative2Zero(bitseq)
    UT.print8bitAlignment(bitseq)

@asyncio.coroutine
def showPN(position=415797, count = 11, numOfShift = 4):
    global rb
    pn = UT.getPN(sync_pn_seed, pnsize)
    numOfPartialPNs = int((pnsize + partialPnSize - 1) / partialPnSize)
    corrarray = np.ndarray(shape=(numOfShift,count))
    corrarray.fill(0.0)
    targetFrameSize = frameSize * numOfPartialPNs

    orgPos = position
    for shift in range(numOfShift):
        position = orgPos + shift
        for idx in range(count):
            source, realPos = yield from rb.readfrom(targetFrameSize, position)
            if (realPos != position):
                print("warning!!  wrong position realPos %d, position %d" % (realPos, position))

            try:
                extractedPN = np.ndarray(shape=(pnsize))
                extractedPN.fill(0.0)
                for pnIdx in range(numOfPartialPNs):
                    begin = pnIdx * frameSize
                    end = begin + frameSize
                    transformed = np.fft.fft(source[begin:end])
                    b, e = UT.getTargetFreqBand(transformed, partialPnSize, subband[0])
                    extractedPN[pnIdx * partialPnSize : (pnIdx + 1) * partialPnSize] = transformed.real[b:e]

                posRead = realPos + int(targetFrameSize)
                r1, r2 = UT.SGN(extractedPN, pn)
                #print("before: ", r1, extractedPN)
                #extractedPN = extractedPN * r1  # pn 사인 원복
                #print("after: ", r1, extractedPN)
                #print (r1)
                corr = abs(np.corrcoef(extractedPN, pn)[0][1])
                corrarray[shift][idx] = corr

            except Exception as err:
                print(err)
                return -1, posRead

            position = posRead + targetFrameSize
    print(corrarray)
    plt.plot(corrarray[:,:])
    plt.show()
    return


@asyncio.coroutine
def findSYNC(position, searchingMaxCorr = True):
    if searchingMaxCorr:
        posMaxCorr, posRead = yield from getMaxCorr(position)
        if (posMaxCorr < 0):
            return posMaxCorr, posRead
    else:
        posMaxCorr = position

    # find next position after SYNC
    numOfPartialPNs = int((pnsize + partialPnSize - 1) / partialPnSize)
    print("begin of SYNC : ", posMaxCorr)
    endOfSync = posMaxCorr + (numOfPartialPNs * NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN * frameSize)
    print("end of SYNC : ", endOfSync)
    return posMaxCorr, endOfSync

def extractPN(source, numOfFramesPerPartialPn, numOfPartialPNs):
    extractedPN = []
    for pnIdx in range(numOfPartialPNs):
        begin = pnIdx * numOfFramesPerPartialPn * frameSize
        end = begin + frameSize
        transformed = np.fft.rfft(source[begin:end])
        freq = [BASE_FREQ_OF_DATA_EMBEDDING + i * FREQ_INTERVAL_OF_DATA_EMBEDDING for i in range(partialPnSize)]
        # print ("bef freq:", freq)
        for fIdx, val in enumerate(freq):
            freq[fIdx] = int(val / (fs / 2 / (len(transformed))))  # search bands
            extractedPN.append(abs(transformed[freq[fIdx]]))
    return extractedPN

@asyncio.coroutine
def findMSG(position, count=20):
    global rb
    idx = 0
    numOfPartialPNs = int(pnsize / partialPnSize)
    pnlist = UT.generatePnList()
    result = str("")
    print("At %d ****" % position, end=" ")
    while(True):
        sizeOfFramesPerFullPN = frameSize * numOfPartialPNs * NUM_OF_FRAMES_PER_PARTIAL_MSG_PN
        begin = position + (idx * sizeOfFramesPerFullPN)
        idx += 1
        Nframes, realPos = yield from rb.readfrom(sizeOfFramesPerFullPN, begin)
        if (realPos != begin):
            print("!!!!! wrong position realPos %d, begin %d" % (realPos, begin))
        end = begin + sizeOfFramesPerFullPN

        # extract PN
        extractedPN = extractPN(Nframes, NUM_OF_FRAMES_PER_PARTIAL_MSG_PN, numOfPartialPNs)

        # compare extracted PN and ascii PNs in list
        maxCorr = 0
        maxCorrIdx = -1
        # coarray = []
        for i, value in enumerate(pnlist):
            corr = abs(np.corrcoef(extractedPN, value)[0][1])
            if corr > maxCorr:
                maxCorr = corr
                maxCorrIdx = i
            # coarray.append(corr)

        # if position>928933:
        #     plt.plot(coarray)
        #     plt.show()

        character = chr(maxCorrIdx)
        print("%c[ %d , %2.2f]" % (character, maxCorrIdx, maxCorr), end=" ")
        #print("%c " % (character), end=" ")

        # if the extracted PN is for ascii '\n', break out
        if character == '\n':
            break
        if idx > count:
            break
        result += character
    print("")
    return result, end


@asyncio.coroutine
def getMaxCorr(position):
    global rb
    found = []
    max_corr = 0
    posMaxCorr = -1
    posRead = 0
    pn = UT.getPN(sync_pn_seed, pnsize)
    numOfPartialPNs = int((pnsize + partialPnSize - 1) / partialPnSize)
    #print("pn: ", pn[:20])
    corrarray = []
    dotparray = []
    aa  = datetime.datetime.now()
    targetFrameSize = frameSize * (numOfPartialPNs * NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN) * 2
    source, realPos = yield from rb.readfrom(targetFrameSize, position)
    # plt.plot(source)
    # plt.show()
    if (realPos != position):
        print("warning!!  wrong position realPos %d, position %d" % (realPos, position))

    try:
        idxList = [ int(i * (frameSize * NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN / 2)) for i in range( numOfPartialPNs * 2)]
        #print("getMaxCorr: pos", position, idxList)
        for idx in idxList: # partialPnSize 의 절반씩 옮기면서 correlation 계산
            extractedPN = []
            for pnIdx in range(numOfPartialPNs):
                begin = idx + pnIdx * NUM_OF_FRAMES_PER_PARTIAL_SYNC_PN * frameSize
                end = begin + frameSize
                transformed = np.fft.rfft(source[begin:end])
                freq = [BASE_FREQ_OF_DATA_EMBEDDING + i * FREQ_INTERVAL_OF_DATA_EMBEDDING for i in range(partialPnSize)]
                #print ("bef freq:", freq)
                for fIdx, val in enumerate(freq):
                    freq[fIdx] = int(val / (fs/2/(len(transformed)))) # search bands
                    extractedPN.append(abs(transformed[freq[fIdx]]))

                #print("aft freq:", freq)
                #b, e = UT.getTargetFreqBand(transformed, partialPnSize, subband[0])

            posRead = realPos + idx + int(numOfPartialPNs * frameSize)
            corr = abs(np.corrcoef(extractedPN, pn)[0][1])
            corrarray.append(corr)

            r1, r2 = UT.SGN(extractedPN, pn)
            dotparray.append(r2)

            if max_corr <= corr:
                max_corr = corr
                posMaxCorr = realPos + idx

    except Exception as err:
        print (err)
        return -1, posRead
    # plt.plot(dotparray)
    # plt.plot(corrarray)
    # plt.show()

    bb = datetime.datetime.now()
    print("execution time: ", (bb-aa).total_seconds())
    print ("--------------------max: %10.2f, %d" % (max_corr, posMaxCorr))
    #print("pn_max/min : ", pn.max(), pn.min())
    # plt.plot(corrarray)
    # plt.show()

    if max_corr >= detectionThreshold:
        print("--------------------Found maximum:", posMaxCorr, max_corr, posRead)
        return posMaxCorr, posRead
    else:
        #print("Can't find cross-correlation peak over 0.8, max:%d in idx:%d" % (max_value, max_index))
        return -1, posRead

import threading
class streamReader(threading.Thread):
    def __init__(self, inStream):
        self.inStream = inStream
        super().__init__()

    def run(self):
        global rb
        q = True
        count = 0
        print("begin")
        while (True):
            try:
                # print("before", end=" ")
                if USE_MIC:
                    data = self.inStream.read(CHUNK)
                    data = np.fromstring(data, np.dtype('int16'))
                    # FS.updateSpectrum(data)
                    data = np.float32(data) / norm_fact[data.dtype.name]
                    # print(len(data))
                    rb.write(data)
                    if (q and rb.writeptr() > RATE * 30):
                        UF.wavwrite(rb.dat()[:RATE * 30], fs, "last_mic_in.wav")
                        q = False
                    # count += 1
                    # if (count % 1 == 0):
                    time.sleep(0.001)
                else:
                    size = CHUNK
                    if (len(self.inStream) < CHUNK):
                        size = len(self.inStream)
                    # FS.updateSpectrum(inStream[0:size], 44100)
                    rb.write(self.inStream[0:size])
                    self.inStream = self.inStream[size:]
                    if len(self.inStream) == 0:
                        raise IOError("end of stream")
                    time.sleep(0.002)
                    # print("Done")
            except IOError as err:
                print(err, datetime.datetime.now())
                if (str(err) == "end of stream"):
                    break
                pass

@asyncio.coroutine
def readStream(inStream):
    global rb
    q = True
    count = 0
    while(True):
        try:
            #print("before", end=" ")
            if USE_MIC:
                if inStream.get_read_available() >= CHUNK: # to avoid blocking of all threads
                    data = inStream.read(CHUNK)
                    inStream.get_read_available()
                    data = np.fromstring(data, np.dtype('int16'))
                    #FS.updateSpectrum(data)
                    data = np.float32(data) / norm_fact[data.dtype.name]
                    #print(len(data))
                    rb.write(data)
                    if (q and rb.writeptr() > RATE * 30):
                        UF.wavwrite(rb.dat()[:RATE * 30], fs, "last_mic_in.wav")
                        q = False
                # count += 1
                # if (count % 1 == 0):
                yield from asyncio.sleep(0.001)
            else:
                size = CHUNK
                if (len(inStream) < CHUNK):
                    size = len(inStream)
                #FS.updateSpectrum(inStream[0:size], 44100)
                rb.write(inStream[0:size])
                inStream = inStream[size:]
                if len(inStream) == 0:
                    raise IOError("end of stream")
                yield from asyncio.sleep(0.002)
            #print("Done")
        except IOError as err:
            print(err, datetime.datetime.now())
            if (str(err) == "end of stream"):
                break
            pass
    #print(len(q), frame_count, time_info["input_buffer_adc_time"], time_info["current_time"], time_info["output_buffer_dac_time"], datetime.datetime.now())

def mic_on():
    if USE_MIC:
        print(sys._getframe().f_code.co_name)
        global p
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK*500)
        stream.start_stream()
        print("* recording")
        return stream
    else:
        print("stream input from the file : ", inputFile)
        input_signal = UT.readWav(inputFile)
        return input_signal



def mic_off(stream = None):
    print(sys._getframe().f_code.co_name)
    if USE_MIC:
        print("* done recording")
        stream.stop_stream()
        stream.close()
        global p
        p.terminate()
    else:
        print()

@asyncio.coroutine
def listen(doFunc):
    print(sys._getframe().f_code.co_name)
    global rb
    time.sleep(1)
    initialTime = datetime.datetime.now()
    lastPosRead = 0
    posRead = 0
    sumPosRead = 0
    #UT.tprint("init: ", initialTime)
    #
    # p = 1121493+3
    # for i in range(16):
    #     print (i-8, p-8+i, end=": ***** ")
    #     yield from printSYNC(p-8+i)
    # #yield from printSYNC(p + 90112)
    # yield from printMSG(p + 90112)
    # return

    while True:
        try:
            msg = ""
            begin = datetime.datetime.now()
            if (begin - initialTime).total_seconds() > TIMEOUT > 0:
                break;
            #posRead = getCurrentPos(begin - initialTime) # absolute position
            sumPosRead += (watingtime * RATE)
            posRead = sumPosRead
            #UT.tprint("begin: ", begin, begin-initialTime)
            if (posRead < lastPosRead):
                posRead = lastPosRead
            #targetStream = inStream[posRead:]  # Update readfrom pointer
            #targetStream = yield from rb.readfrom(posRead, frameSize*2+2, posRead)
            print ("pos: %10.4f, %d" % ((posRead / fs), posRead))
            posFound, posRead = yield from findSYNC(posRead, True)
            if posFound < 0:
                raise NotImplementedError("SYN is not found")

            # Search Message
            msg, tmpPosRead = yield from findMSG(posRead)
            # posRead = 928592 - frameSize * 1
            # for iii in range(12):
            #     print("---------", iii, posRead + iii * frameSize/3)
            #     msg, tmpPosRead = yield from findMSG(posRead + iii * frameSize/3, 20)
            if (len(msg) <= 0):
                raise NotImplementedError("MSG is not found")
            lastPosRead = tmpPosRead
            doFunc(msg)
        except NotImplementedError as err:
            print(err)
            #q = 0 # Do nothing
        except Exception as err:
            print(err, err.__traceback__)
        finally:
            end = datetime.datetime.now()
            diff_sec = watingtime - (end-begin).total_seconds()
            if (diff_sec > 0):
                print("wait", diff_sec/2, begin, end)
                yield from asyncio.sleep(diff_sec/2)
            else:
                yield from asyncio.sleep(0.005)
    return

def doMsg(msg):
    chrome_path = 'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe %s'
    #webbrowser.get('google-chrome').open(msg)
    subprocess.call(["C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", msg])

def getCurrentPos(duration):
    duration_in_sample = int(fs * duration.total_seconds())
    return duration_in_sample

if __name__ == "__main__":
    rb = UT.RIngBuffer(RATE * 60)
    inStream = mic_on()
    loop = asyncio.get_event_loop()
    # srThread = streamReader(inStream)
    # srThread.start()
    # print("sr start")

    loop.run_until_complete(asyncio.gather(
        readStream(inStream),
        listen(doMsg),
    ))

    # srThread.join()
    mic_off(inStream)
