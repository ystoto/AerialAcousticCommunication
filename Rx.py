# function to call the main analysis/synthesis functions in software/models/stft.py
import numpy as np
import os, sys
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
from wm_util import pnsize, frameSize, sync_pn_seed, msg_pn_seed, fs, SYNC, NUMOFSYNCREPEAT, detectionThreshold,\
    subband, norm_fact

#inputFile = 'last_mic_in.wav'
inputFile = 'SleepAway_partial_stft.wav'
USE_MIC = False
watingtime = 1.4

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
        begin, end = UT.getTargetFreqBand(transformed, subband[0])
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

async def getProperPN(position, left_pnseed, right_pnseed):
    frame, realPos = await rb.read(frameSize, position)
    if (realPos != position):
        print("!!!!! wrong position realPos %d, begin %d" % (realPos, position))
        return -1
    transformed = MDCT.mdct(frame)

    leftPN = UT.getPN(left_pnseed, pnsize)
    rightPN = UT.getPN(right_pnseed, pnsize)
    for num, band in enumerate(subband):
        begin, end = UT.getTargetFreqBand(transformed, band)
        a = abs(np.corrcoef(transformed[begin:end, 1], leftPN)[0][1])
        b = abs(np.corrcoef(transformed[begin:end, 1], rightPN)[0][1])
        if a > b:
            return left_pnseed
        else:
            return right_pnseed

async def extractBitSequence(position, seed, count=520, endOfSequence = ''):
    global rb
    #result = np.ndarray(shape=(len(subband),1))
    result = [[] for i in range(len(subband))]
    result2 = [[] for i in range(len(subband))]
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
        pn2 = UT.getPN(msg_pn_seed, pnsize)
        transformed = MDCT.mdct(frame)
        for num, band in enumerate(subband):
            begin, end = UT.getTargetFreqBand(transformed, band)
            result[num].append(UT.SGN(transformed[begin:end, 1], pn))
            #result[num].append(abs(np.corrcoef(transformed[begin:end, 1], pn)[0][1]))
            result2[num].append(abs(np.corrcoef(transformed[begin:end, 1], pn2)[0][1]))
            # result[num].append(np.correlate(transformed[begin:end, 1], pn))
            # result2[num].append(np.correlate(transformed[begin:end, 1], pn2))
        idx += 1
        if (count > 0 and idx >= count):
            break
        if (idx % 8 == 0 and len(e) == 8):
            if (e == bitarray.bitarray(UT.convertNegative2Zero(result[0][-8:]))):
                break
    return result, end, result2

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

    for ii in range(1):
        p = posMaxCorr+ii
        print ("---", p, (p - 220500)/1024)
        bitsSequence, tmpPosRead, tmp = await extractBitSequence(p, sync_pn_seed, len(SYNC))
        UT.printWithIndex("SYNC: ", bitsSequence[0], True)
        UT.printWithIndex("DATA: ", tmp[0])
        UT.printwithsign("SYNC: ", SYNC)
        UT.printwithsign("BIT : ", bitsSequence[0])
    #bitsSequence, tmpPosRead, tmp = await extractBitSequence(posMaxCorr, sync_pn_seed, len(SYNC))
    numberOfMatchedTailBits = matchBitSequence(bitsSequence[0], SYNC)
    if (numberOfMatchedTailBits <= 0):
        return -1, posRead

    # Confirm whole SYNC
    numberOfHeadBits = len(SYNC) - numberOfMatchedTailBits
    position = posMaxCorr - (frameSize * numberOfHeadBits)
    wholeBits, tmpPosRead, tmp = await extractBitSequence(position, sync_pn_seed, len(SYNC))
    tmpNumberOfMatchedBits = matchBitSequence(wholeBits[0], SYNC)
    if tmpNumberOfMatchedBits < len(SYNC):
        print("Not matched whole SYNC at %d" % position)
        UT.printwithsign("SYNC: ", SYNC)
        UT.printwithsign("BIT : ", wholeBits[0])
        return -1, posRead

    print("whole SYNC are matched at %d" % position)
    UT.printwithsign("SYNC: ", SYNC)
    UT.printwithsign("BIT : ", wholeBits[0])
    return posMaxCorr, posMaxCorr + (frameSize * numberOfMatchedTailBits)

async def findMSG(position):
    #print(sys._getframe().f_code.co_name)
    bitsSequence, posRead, tmp = await extractBitSequence(position, msg_pn_seed, endOfSequence='\n')
    for i in range(len(bitsSequence)):
        UT.convertNegative2Zero(bitsSequence[i])
        ba = bitarray.bitarray(bitsSequence[i])
        UT.print8bitAlignment(bitsSequence[i])
        msg = ba.tostring().replace('\n', '')
        print(msg)
    return msg, posRead

async def getMaxCorr(position):
    global rb
    found = []
    max_corr = 0
    posMaxCorr = -1
    posRead = 0
    pn = UT.getPN(sync_pn_seed, pnsize)
    #print("pn: ", pn[:20])
    corrarray = []
    aa  = datetime.datetime.now()
    source, realPos = await rb.read(frameSize * 2 + 8, position)
    if (realPos != position):
        print("!!!!! wrong position realPos %d, position %d" % (realPos, position))
    for idx in range(frameSize + 9):
        transformed = MDCT.mdct(source[idx:idx + int(frameSize)])
        posRead = realPos + idx + int(frameSize)
        begin, end = UT.getTargetFreqBand(transformed, subband[0])
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

    if max_corr > detectionThreshold:
        print("--------------------Found maximum:", posMaxCorr, max_corr, posRead)
        return posMaxCorr, posRead
    else:
        #print("Can't find cross-correlation peak over 0.8, max:%d in idx:%d" % (max_value, max_index))
        return -1, posRead

async def readStream(inStream):
    global rb
    q = True
    count = 0
    while(True):
        try:
            #print("before", end=" ")
            if USE_MIC:
                data = inStream.read(CHUNK)
                data = np.fromstring(data, np.dtype('int16'))
                data = np.float32(data) / norm_fact[data.dtype.name]
                #print(len(data))
                rb.write(data)
                if (q and rb.writeptr() > RATE * 20):
                    UF.wavwrite(rb.dat()[:RATE * 20], fs, "last_mic_in.wav")
                    q = False
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
            if posFound < 0:
                raise NotImplementedError("SYN is not found")

            # Search remain SYNCs
            found = 0
            for idx in range(NUMOFSYNCREPEAT):
                position = posRead + idx * frameSize * len(SYNC)
                seed = await getProperPN(position, sync_pn_seed, msg_pn_seed)
                if seed == msg_pn_seed:
                    posRead = position
                    found = 1
                    break
            if found == 0:
                raise NotImplementedError("MSG corr is not found")

            msg, tmpPosRead = await findMSG(posRead)
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
                await asyncio.sleep(diff_sec)
            else:
                await asyncio.sleep(0.005)
    return

def doMsg(msg):
    chrome_path = 'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe %s'
    #webbrowser.get('google-chrome').open(msg)
    subprocess.call(["C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", msg])

if __name__ == "__main__":
    # invereSYNC = [i* -1 for i in SYNC]
    # UT.printwithsign("", invereSYNC)
    # UT.printwithsign("", SYNC)
    # print ("match", matchBitSequence(invereSYNC, SYNC))
    rb = UT.RIngBuffer(RATE * 60)
    inStream = mic_on()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(
        readStream(inStream),
        listen(doMsg),
    ))

    mic_off(inStream)

