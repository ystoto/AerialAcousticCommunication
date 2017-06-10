# function to call the main analysis/synthesis functions in software/models/stft.py
import numpy as np
import numpy.random as PN
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import get_window
import bitarray
import datetime
import time
import subprocess
import pyaudio

#os.environ['PYTHONASYNCIODEBUG'] = '1'
import asyncio

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'E:/Dropbox/sms/software/models'))
#from .stft import stftAnal, stftSynth

import importlib.util
spec = importlib.util.spec_from_file_location("stft", "E:/Dropbox/sms/software/models/stft.py")
STFT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STFT)

import utilFunctions as UF
import mdct as MDCT

import wm_util as UT
from wm_util import tprint

inputFile = 'SleepAway_stft.wav'
USE_MIC = True
fs = 44100
frameSize = 1024
pnsize = 32
sync_pn_seed = 1
data_pn_seed = 2
watingtime = 1.4
#SYNC = [+1]
SYNC = [-1,+1,+1,+1,-1,+1,-1,+1,-1,-1,
        +1,-1,-1,+1,+1,+1,-1,+1,-1,+1,
        -1,-1,+1,-1,+1,-1,-1,+1,+1,-1,
        +1,-1,-1,+1,+1,-1,+1,+1,+1,+1,
        +1,+1,-1,+1,+1,+1,-1,+1,-1,+1,
        +1,-1,-1,-1,-1,-1,-1,-1,+1,+1,
        +1,-1,-1,-1]
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
TIMEOUT = 0



def extractSYNC(target):
    found = []
    for idx, value in enumerate(SYNC):
        frame = target[idx * frameSize : (idx + 1) * frameSize]
        pn = UT.getPN(sync_pn_seed, pnsize)
        transformed = MDCT.mdct(frame)
        begin, end = UT.getTargetFreqBand(transformed)
        result = UT.SGN(transformed[begin:end, 1], pn)
        if (result == value):
            found.append(result)
        else:
            break

    if (found == SYNC):
        print("SYNC is found")
        return True
    else:
        print("SYNC is not found: ", found)
        return False

async def extractBitSequence(position, seed, count=520, endOfSequence = ''):
    global rb
    #result = np.ndarray(shape=(len(UT.subband),1))
    result = [[] for i in range(len(UT.subband))]
    idx = 0
    end = 0
    e = bitarray.bitarray()
    e.fromstring(endOfSequence)
    if (len(e) != 0 and len(e) != 8):
        print("Wrong endOfSequence, size of e:", len(e))
        return -1, -1

    while(True):
        begin = position + (idx * frameSize)
        frame, realPos = await rb.read(frameSize, begin)
        if (realPos != begin):
            print("!!!!! wrong position realPos %d, begin %d" % (realPos, begin))
        end = begin + frameSize
        pn = UT.getPN(seed, pnsize)
        transformed = MDCT.mdct(frame)
        for num, band in enumerate(UT.subband):
            begin, end = UT.getTargetFreqBand(transformed, band)
            value = UT.SGN(transformed[begin:end, 1], pn)
            result[num].append(value)
        idx += 1
        if (count > 0 and idx >= count):
            break
        if (idx % 8 == 0 and len(e) == 8):
            if (e == bitarray.bitarray(UT.convertNegative2Zero(result[0][-8:]))):
                break
    return result, end

def matchBitSequence(left, right):
    len_L = len(left)
    len_R = len(right)
    for idx in range(len_L):
        if (left[:len_L - idx] == right[idx:]):
            return len_L - idx
    return 0

async def findSYNC(position):
    posMaxCorr, posRead = await getMaxCorr(position)
    if (posMaxCorr < 0):
        return posMaxCorr, posRead

    bitsSequence, tmpPosRead = await extractBitSequence(posMaxCorr, sync_pn_seed, len(SYNC))
    UT.printwithsign("SYNC: ", SYNC)
    UT.printwithsign("BIT : ", bitsSequence[0])
    matched_bits = matchBitSequence(bitsSequence[0], SYNC)
    if (matched_bits <= 0):
        return -1, posRead

    return posMaxCorr, posMaxCorr + (frameSize * matched_bits)

async def getMaxCorr(position):
    global rb
    found = []
    max_corr = 0
    posMaxCorr = -1
    posRead = 0
    pn = UT.getPN(sync_pn_seed, pnsize) * 100
    #print("pn: ", pn[:20])
    corrarray = []
    aa  = datetime.datetime.now()
    source, realPos = await rb.read(frameSize * 2, position)
    if (realPos != position):
        print("!!!!! wrong position realPos %d, position %d" % (realPos, position))
    for idx in range(frameSize + 1):
        transformed = MDCT.mdct(source[idx:idx + int(frameSize)])
        posRead = realPos + idx + int(frameSize)
        begin, end = UT.getTargetFreqBand(transformed)
        corr = abs(np.corrcoef(transformed[begin:end, 1], pn)[0][1])
        #corr = np.correlate(transformed[begin:end, 1], pn)
        corrarray.append(corr)
        if max_corr <= corr:
            max_corr = corr
            posMaxCorr = realPos + idx

    bb = datetime.datetime.now()
    #print("execution time: ", bb-aa)
    print ("--------------------max: ", max_corr, posMaxCorr)
    #print("pn_max/min : ", pn.max(), pn.min())
    # plt.plot(corrarray)
    # plt.show()

    if max_corr > UT.detectionThreshold:
        print("--------------------Found maximum:", posMaxCorr, max_corr, posRead)
        return posMaxCorr, posRead
    else:
        #print("Can't find cross-correlation peak over 0.8, max:%d in idx:%d" % (max_value, max_index))
        return -1, posRead

async def readStream(inStream):
    global rb
    # q = True
    count = 0
    while(True):
        try:
            #print("before", end=" ")
            if USE_MIC:
                data = inStream.read(CHUNK)
                data = np.fromstring(data, np.dtype('int16'))
                data = np.float32(data) / UT.norm_fact[data.dtype.name]
                #print(len(data))
                rb.write(data)
                # if (q and rb.writeptr() > RATE * 10):
                #     UF.wavwrite(rb.dat()[:220500*2], fs, "SleepAway_mic2.wav")
                #     q = False
                # count += 1
                # if (count % 1 == 0):
                await asyncio.sleep(0.001)
            else:
                size = CHUNK
                if (len(inStream) < CHUNK):
                    size = len(inStream)
                rb.write(inStream[0:size])
                inStream = inStream[size:]
                if len(inStream) == 0:
                    raise IOError("end of stream")
                await asyncio.sleep(0.002)
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

async def findMSG(position):
    #print(sys._getframe().f_code.co_name)
    bitsSequence, posRead = await extractBitSequence(position, data_pn_seed, endOfSequence='\n')
    for i in range(len(bitsSequence)):
        UT.convertNegative2Zero(bitsSequence[i])
        ba = bitarray.bitarray(bitsSequence[i])
        UT.print8bitAlignment(bitsSequence[i])
        msg = ba.tostring().replace('\n', '')
        print(msg)
    return msg, posRead

def mic_off(stream = None):
    print(sys._getframe().f_code.co_name)
    if USE_MIC:
        print("* done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()
    else:
        print()

def getCurrentPos(duration):
    duration_in_sample = int(fs * duration.total_seconds())
    return duration_in_sample

async def listen(doFunc):
    print(sys._getframe().f_code.co_name)
    global rb
    time.sleep(1)
    initialTime = datetime.datetime.now()
    lastPosRead = 0
    posRead = 0
    #UT.tprint("init: ", initialTime)
    while True:
        try:
            msg = ""
            begin = datetime.datetime.now()
            if (begin - initialTime).total_seconds() > TIMEOUT > 0:
                break;
            posRead = getCurrentPos(begin - initialTime) # absolute position
            #UT.tprint("begin: ", begin, begin-initialTime)
            if (posRead < lastPosRead):
                posRead = lastPosRead
            #targetStream = inStream[posRead:]  # Update read pointer
            #targetStream = await rb.read(posRead, frameSize*2+2, posRead)
            print ("pos: ", posRead / fs)
            posFound, posRead = await findSYNC(posRead)

            if (posFound < 0):
                raise NotImplementedError("SYN not found")
            msg, tmpPosRead = await findMSG(posRead)
            if (len(msg) <= 0):
                raise NotImplementedError("MSG not found")
            lastPosRead = tmpPosRead
            doFunc(msg)
        except NotImplementedError:
            q = 0 # Do nothing
        except Exception as err:
            print(err, err.__traceback__)
        finally:
            end = datetime.datetime.now()
            diff_sec = watingtime - (end-begin).total_seconds()
            if (diff_sec > 0):
                await asyncio.sleep(diff_sec)
            else:
                await asyncio.sleep(0.005)
    return

def doMsg(msg):
    #chrome_path = 'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe %s'
    #webbrowser.get('google-chrome').open(msg)
    subprocess.call(["C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", msg])

if __name__ == "__main__":
    rb = UT.RIngBuffer(RATE * 60)
    inStream = mic_on()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(
        readStream(inStream),
        listen(doMsg),
    ))

    mic_off(inStream)