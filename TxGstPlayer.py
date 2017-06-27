# import gi
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst
# import time
# import asyncio
#
# VIDEO_PATH = "C:\\Users\\Public\\Videos\\Sample Videos\\SleepAway_stft.mp4"
#
#
# if __name__ == "__main__":
#     Gst.init(None)
#     pipeline = Gst.parse_launch("playbin uri=C:\\Users\\Public\\Videos\\Sample\ Videos\\SleepAway_stft.mp4")
#     pipeline.set_state(Gst.State.PLAYING)
#     bus = pipeline.get_bus()
#     msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
#     time.sleep(1000)

####################################################################################

# !/usr/bin/env python
import os
import sys
import ctypes
import pyaudio
import gi
import numpy as np
import wave
import struct
import datetime
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, Gtk
from gi.repository import GdkX11, GstVideo

import asyncio
import Tx
import wm_util as UT
from wm_util import norm_fact

import time
from subprocess import call
VIDEO_PATH = "SleepAway_stft.mp4"
PLAYING = 0
PAUSED = 1
STOPPED = 2
bands_range = [29, 59, 119, 237, 474, 947, 1889, 3770, 7523, 15011]

class GTK_Main(object):
    def __init__(self):
        self.play_status = STOPPED
        self.IS_GST010 = Gst.version()[0] == 0
        self.volume = 100
        window = Gtk.Window(Gtk.WindowType.TOPLEVEL)
        window.set_title("WM-Player")
        window.set_default_size(450, -1)
        window.connect("destroy", Gtk.main_quit, "WM destroy")

        self._init_audio_buffer()

        vbox = Gtk.VBox()
        #vbox = Gtk.Box(Gtk.Orientation.HORIZONTAL, 0)
        vbox.set_margin_top(3)
        vbox.set_margin_bottom(3)
        window.add(vbox)


        # input target media file
        hbox_0st_line = Gtk.HBox()
        vbox.pack_start(hbox_0st_line, False, False, 0)

        self.label_target = Gtk.Label(label='target file')
        self.label_target.set_margin_left(6)
        self.label_target.set_margin_right(6)
        hbox_0st_line.pack_start(self.label_target, False, False, 0)
        self.entry_target = Gtk.Entry()
        self.entry_target.set_text(VIDEO_PATH)
        hbox_0st_line.add(self.entry_target)


        hbox_1st_line = Gtk.HBox()
        vbox.pack_start(hbox_1st_line, False, False, 0)
        # self.entry = Gtk.Entry()
        # hbox.add(self.entry)

        # play button
        self.playButtonImage = Gtk.Image()
        self.playButtonImage.set_from_stock("gtk-media-play", Gtk.IconSize.BUTTON)
        self.playButton = Gtk.Button.new()
        self.playButton.add(self.playButtonImage)
        self.playButton.connect("clicked", self.playToggled)
        hbox_1st_line.pack_start(self.playButton, False, False, 0)

        # stop button
        self.stopButtonImage = Gtk.Image()
        self.stopButtonImage.set_from_stock("gtk-media-stop", Gtk.IconSize.BUTTON)
        self.stopButton = Gtk.Button.new()
        self.stopButton.add(self.stopButtonImage)
        self.stopButton.connect("clicked", self.stopToggled)
        hbox_1st_line.pack_start(self.stopButton, False, False, 0)

        # seek to given position
        self.seek_entry = Gtk.Entry()
        hbox_1st_line.add(self.seek_entry)
        self.seekButtonImage = Gtk.Image()
        self.seekButtonImage.set_from_stock("gtk-jump-to", Gtk.IconSize.BUTTON)
        self.seekButton = Gtk.Button.new()
        self.seekButton.add(self.seekButtonImage)
        self.seekButton.connect("clicked", self.seekToggled)
        hbox_1st_line.pack_start(self.seekButton, False, False, 0)
        #hbox_1st_line.add(self.seekButton)

        # seek slider
        hbox_2nd_line = Gtk.HBox()
        vbox.pack_start(hbox_2nd_line, False, False, 0)

        self.label_progress = Gtk.Label(label='progress')
        self.label_progress.set_margin_left(6)
        self.label_progress.set_margin_right(6)
        hbox_2nd_line.pack_start(self.label_progress, False, False, 0)

        self.progress_slider = Gtk.HScale()
        self.progress_slider.set_margin_left(6)
        self.progress_slider.set_margin_right(6)
        self.progress_slider.set_draw_value(False)
        self.progress_slider.set_range(0, 100)
        self.progress_slider.set_increments(1, 10)
        hbox_2nd_line.pack_start(self.progress_slider, True, True, 0)

        self.progress_label = Gtk.Label(label='0:00')
        self.progress_label.set_margin_left(6)
        self.progress_label.set_margin_right(6)
        hbox_2nd_line.pack_start(self.progress_label, False, False, 0)


        # # volume slider
        hbox_3rd_line = Gtk.HBox()
        vbox.pack_start(hbox_3rd_line, False, False, 0)

        self.volume_label = Gtk.Label(label='volume   ')
        self.volume_label.set_margin_left(6)
        self.volume_label.set_margin_right(6)
        hbox_3rd_line.pack_start(self.volume_label, False, False, 0)

        self.volume_slider = Gtk.HScale()
        self.volume_slider.set_margin_left(6)
        self.volume_slider.set_margin_right(6)
        self.volume_slider.set_draw_value(False)
        self.volume_slider.set_range(0, 100)
        self.volume_slider.set_increments(1, 10)
        self.volume_slider.set_value(self.volume)
        self.volume_slider.connect("value-changed", self.volume_changed_cb)
        hbox_3rd_line.pack_start(self.volume_slider, True, True, 0)

        self.volume_value = Gtk.Label(label='0')
        self.volume_value.set_margin_left(6)
        self.volume_value.set_margin_right(6)
        self.volume_value.set_text(str(self.volume))
        hbox_3rd_line.pack_start(self.volume_value, False, False, 0)

        # equalizer preset combobox
        hbox_4rd_line = Gtk.Box()
        vbox.pack_start(hbox_4rd_line, False, False, 0)

        self.eq_label = Gtk.Label(label='equalizer')
        self.eq_label.set_margin_left(6)
        self.eq_label.set_margin_right(6)
        hbox_4rd_line.pack_start(self.eq_label, False, False, 0)

        self.eq_textbox = Gtk.ComboBoxText()
        self.eq_bands_dict = loadEqPresetFile()
        for key in self.eq_bands_dict.keys():
            self.eq_textbox.append(key, key)
        self.eq_textbox.set_valign(Gtk.Align.CENTER)
        self.eq_textbox.connect("changed", self.eq_changed_cb)
        hbox_4rd_line.pack_start(self.eq_textbox, False, False, 0)

        self.eq_slider = [Gtk.VScale() for i in range(10)]
        for i in range(10):
            self.eq_slider[i].set_draw_value(True)
            self.eq_slider[i].set_value_pos(Gtk.PositionType.BOTTOM)
            self.eq_slider[i].set_size_request(18,200)
            self.eq_slider[i].set_range(-24, +12)
            self.eq_slider[i].set_inverted(True)
            hbox_4rd_line.pack_start(self.eq_slider[i], True, True, 0)


        hbox_5th_line = Gtk.HBox()
        vbox.pack_start(hbox_5th_line , False, False, 0)

        # seek to given position
        self.label_wm = Gtk.Label(label='watermark msg')
        self.label_wm.set_margin_left(6)
        self.label_wm.set_margin_right(6)
        hbox_5th_line.pack_start(self.label_wm, False, False, 0)
        self.wm_entry = Gtk.Entry()
        self.wm_entry.set_text("www.naver.com")
        hbox_5th_line.add(self.wm_entry)
        self.wmButtonImage = Gtk.Image()
        self.wmButtonImage.set_from_stock("gtk-go-down", Gtk.IconSize.BUTTON)
        self.wmButton = Gtk.Button.new()
        self.wmButton.add(self.wmButtonImage)
        self.wmButton.connect("clicked", self.wmToggled)
        hbox_5th_line.pack_start(self.wmButton, False, False, 0)
        #hbox_1st_line.add(self.seekButton)


        self.movie_window = Gtk.DrawingArea()
        vbox.add(self.movie_window)
        window.show_all()

        self.player = Gst.ElementFactory.make("playbin", "player")
        vsink = Gst.ElementFactory.make("autovideosink", "vsink")
        self.player.set_property("video-sink", vsink)
        self.pyaudio = None
        asink = self._get_audiosink_bin()
        self.player.set_property("audio-sink", asink)
        bus = self.player.get_bus()
        bus.add_signal_watch()
        bus.enable_sync_message_emission()
        bus.connect("message", self.on_message)
        bus.connect("sync-message::element", self.on_sync_message)

    def __exit__(self, exc_type, exc_value, traceback):
        for file in self.files:
            # self.before_wav.close()
            self.after_wav.close()
            print("b / a wav clsosed")

    def _init_audio_buffer(self):
        self.audio_dump_cnt = 0
        self.src = UT.RIngBuffer(44100 * 30)
        self.sink = UT.RIngBuffer(44100 * 30)
        self.thread = Tx.Start(self.src, self.sink)  # TODO: 1.2 second

    def _get_audiosink_bin(self):
        self.asink_bin = Gst.Bin.new('asinkbin')

        self.eq = Gst.ElementFactory.make("equalizer-10bands", "eq")
        # g_object_set(G_OBJECT(equalizer), "band1", (gdouble) - 24.0, NULL);
        # g_object_set(G_OBJECT(equalizer), "band2", (gdouble) - 24.0, NULL);

        self.vol = Gst.ElementFactory.make("volume", "vol")

        self.asink = Gst.ElementFactory.make("fakesink", "asink")
        self.asink.set_property("signal-handoffs", 1)
        sigid = self.asink.connect('handoff', self.handoff_cb)
        print("sigid:", sigid)

        self.asink_bin.add(self.eq)
        self.asink_bin.add(self.vol)
        self.asink_bin.add(self.asink)

        self.eq.link(self.vol)
        self.vol.link(self.asink)

        gp = Gst.GhostPad.new('sink', self.eq.get_static_pad('sink'))
        self.asink_bin.add_pad(gp) # Only avaiable after bin.add(eq)
        return self.asink_bin

    def eq_changed_cb(self, combobox):
        print("****", sys._getframe().f_code.co_name, combobox.get_active_id())
        bands = self.eq_bands_dict[combobox.get_active_id()]
        for i in range(10):
            self.eq_slider[i].set_value(bands[i])
            self.eq.set_property("band%d" % i, bands[i])
        print ("eq_changed: ", bands)

    def volume_changed_cb(self, gst_range):
        print("****", sys._getframe().f_code.co_name, " - volume: ", int(gst_range.get_value()))
        self.vol.set_property("volume", gst_range.get_value() / 100.0)

    def handoff_cb(self, element, buffer, pad):
        self.audio_dump_cnt += 1
        if self.pyaudio == None:
            in_format, in_rate, in_channels, in_type = getAudioInfo(pad.get_current_caps())
            print(pad.get_current_caps().to_string())
            print ("audio format: %d, rate: %d,  ch: %d,   type: %s " % (in_format, in_rate, in_channels, in_type))
            self.pyaudio = pyaudio.PyAudio()
            self.stream = self.pyaudio.open(format=self.pyaudio.get_format_from_width(in_format),
                            channels=in_channels,
                            rate=in_rate,
                            output=True)
            if in_type.find('f') >= 0:
                audio_dtype = [None, np.float, np.float16, None, np.float32]
            elif in_type.find('s') >= 0:
                audio_dtype = [None, np.int8, np.int16, None, np.int32]
            else:
                audio_dtype = [None, np.uint8, np.uint16, None, np.uint32]
            dt = np.dtype(audio_dtype[in_format])

            if in_type.find("le"):
                self.audio_dtype = dt.newbyteorder('<')
            else:
                self.audio_dtype = dt.newbyteorder('>')
            print("audio_dtype : ", self.audio_dtype)

            # self.before_wav = wave.open("before.wav", "wb")
            # self.before_wav.setparams((in_channels, in_format, in_rate, 0, 'NONE', 'not compressed'))
            self.after_wav = wave.open("after.wav", "wb")
            self.after_wav.setparams((in_channels, in_format, in_rate, 0, 'NONE', 'not compressed'))

        (ret, info) = buffer.map(Gst.MapFlags.READ)
        if ret == True:
            # TODO: wav dump -  wave.py only support integer value not floating point.
            rawdata = np.frombuffer(info.data, dtype=self.audio_dtype)
            # self.before_wav.writeframesraw(np.int32(rawdata * norm_fact['int32']).tobytes())
            org_type_name = rawdata.dtype.name
            normalized_rawdata = np.float32(rawdata) / norm_fact[org_type_name] # normalize rawdata , -1 to 1
            # print("IN ", normalized_rawdata[:10], normalized_rawdata[-10:], normalized_rawdata.dtype, type(normalized_rawdata))
            self.src.write(normalized_rawdata)
            watermarked_data = self.sink.read(rawdata.size)
            watermarked_data *= norm_fact[org_type_name]
            watermarked_data = watermarked_data.astype(dtype=self.audio_dtype, copy=False)
            # print("OU ", watermarked_data[:30], watermarked_data[-30:], watermarked_data.dtype, type(watermarked_data))
            self.stream.write(watermarked_data.tobytes())
            # todo: change the formula according to the input format
            #self.after_wav.writeframesraw(np.int32(watermarked_data * norm_fact['int32']).tobytes())
            #self.after_wav.writeframesraw(np.int16(watermarked_data).tobytes())
        #print ("output data: ", ret, buffer.pts, info.size)

    def _on_video_realize(self, widget):
        # The window handle must be retrieved first in GUI-thread and before
        # playing pipeline.
        video_window = self.movie_window.get_property('window')
        if sys.platform == "win32":
            if not video_window.ensure_native():
                print("Error - video playback requires a native window")
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object]
            drawingarea_gpointer = ctypes.pythonapi.PyCapsule_GetPointer(video_window.__gpointer__, None)
            gdkdll = ctypes.CDLL("libgdk-3-0.dll")
            self._video_window_handle = gdkdll.gdk_win32_window_get_handle(drawingarea_gpointer)
            print("111")
            #widget.set_window_handle(self._video_window_handle)
        else:
            self._video_window_handle = video_window.get_xid()
            print("222")

    def on_message(self, bus, message):
        #print("on_message")
        t = message.type
        if t == Gst.MessageType.EOS:
            self.player.set_state(Gst.State.NULL)
            self.stopToggled(None)
        elif t == Gst.MessageType.ERROR:
            self.player.set_state(Gst.State.NULL)
            err, debug = message.parse_error()
            print("Error: %s" % err, debug)
            self.stopToggled(None)

        self.updateButtons()

    def on_sync_message(self, bus, message):
        if message.get_structure().get_name() == 'prepare-window-handle':
            imagesink = message.src
            imagesink.set_property("force-aspect-ratio", True)
            #imagesink.set_window_handle(self.movie_window.get_property('window').get_xid())
            #self._on_video_realize(imagesink)

    # def on_finished(self, player):
    #     self.play_status = STOPPED
    #     self.progress_slider.set_value(0)
    #     self.progress_label.set_text("0:00")
    #     self.updateButtons()

    def play(self):
        filepath = self.entry_target.get_text().strip()#VIDEO_PATH  # self.entry.get_text().strip()
        if len(filepath) <= 0:
            filepath = VIDEO_PATH

        if os.path.isfile(filepath):
            filepath = os.path.realpath(filepath)
            self.player.set_property("uri", "file:///" + filepath)
            self.player.set_state(Gst.State.PLAYING)
            GObject.timeout_add(1000, self.updateProgressSlider)
            print(self.player.get_property("uri"))
            print("START")

    def stop(self):
        self.player.set_state(Gst.State.NULL)

    def pause(self):
        self.player.set_state(Gst.State.PAUSED)

    def seekToggled(self, w):
        pos =self.seek_entry.get_text().strip()
        if len(pos) <= 0:
            self.seek_entry.set_text("")
            return

        pos = int(pos)
        duration_nanosecs = self.player.query_duration(Gst.Format.TIME)[1]
        duration = float(duration_nanosecs) / Gst.SECOND
        if pos >= duration - 5:
            pos = duration - 5

        if pos < 0:
            pos = 0
        print ("pos: ", pos, pos*Gst.SECOND)
        pos_ns = pos * Gst.SECOND
        self.player.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, pos_ns)

    def stopToggled(self, w):
        self.progress_slider.set_value(0)
        self.progress_label.set_text("0:00")
        self.stop()
        self.play_status = STOPPED
        self.updateButtons()
        if self.after_wav is not None:
            self.after_wav.close()

    def playToggled(self, w):
        if self.play_status == STOPPED or self.play_status == PAUSED:
            self.play()
            self.play_status = PLAYING
        else:
            self.pause()
            self.play_status = PAUSED

        self.updateButtons()

    def wmToggled(self, w):
        msg = self.wm_entry.get_text().strip()
        msg += '\n'
        self.thread.requestWM(msg)

    def updateProgressSlider(self):
        if self.play_status == STOPPED:
            return False  # cancel timeout

        try:
            nanosecs = self.player.query_position(Gst.Format.TIME)[1]
            duration_nanosecs = self.player.query_duration(Gst.Format.TIME)[1]

            # block seek handler so we don't seek when we set_value()
            # self.slider.handler_block_by_func(self.on_slider_change)

            duration = float(duration_nanosecs) / Gst.SECOND
            position = float(nanosecs) / Gst.SECOND
            #print("prog:", duration, position)
            self.progress_slider.set_range(0, duration)
            self.progress_slider.set_value(position)
            self.progress_label.set_text("%d" % (position / 60) + ":%02d" % (position % 60))
            # self.slider.handler_unblock_by_func(self.on_slider_change)
        except Exception as e:
            # pipeline must not be ready and does not know position
            print(e)
            pass
        return True

    def updateButtons(self):
        if self.play_status == STOPPED or self.play_status == PAUSED:
            self.playButtonImage.set_from_stock("gtk-media-play", Gtk.IconSize.BUTTON)
        else:
            self.playButtonImage.set_from_stock("gtk-media-pause", Gtk.IconSize.BUTTON)

def loadEqPresetFile():
    f = open("equalizer_preset.txt", "r")
    result = dict()
    while True:
        line = f.readline()
        if not line or len(line) == 0:
            break

        if line[0] == '[':
            begin = 1
            end = line.rfind(']')
            key = line[begin:end]
            bands = []
            for i in range(10):
                line = f.readline()
                begin = line.rfind('=') + 1
                bands.append(float(line[begin:]))
            result[key] = bands
    f.close()
    return result

def getAudioInfo(caps):
    structure = caps.get_structure(0)
    ret, channels = structure.get_int("channels")
    ret, rate = structure.get_int("rate")
    type = structure.get_string("format")
    if type.find("32") >= 0:
        format = 4
    elif type.find("24") >= 0:
        format = 3
    elif type.find("16") >= 0:
        format = 2
    else:
        format = 1

    return format, rate, channels, type.lower()


if __name__ == "__main__":
    # call(["gst-launch", "playbin", "uri=\"file:///E:\\\\PycharmProjects\\\\AerialAcousticCommunication\\\\Kalimba.mp3\""])
    # time.sleep(1000)

    GObject.threads_init()
    Gst.init(None)
    obj = GTK_Main()
    Gtk.main()