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
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, Gtk
from gi.repository import GdkX11, GstVideo

import time
from subprocess import call
VIDEO_PATH = "SleepAway_stft.mp4"

class GTK_Main(object):
    def __init__(self):
        window = Gtk.Window(Gtk.WindowType.TOPLEVEL)
        window.set_title("Audio-Player")
        window.set_default_size(300, -1)
        window.connect("destroy", Gtk.main_quit, "WM destroy")

        vbox = Gtk.VBox()
        window.add(vbox)
        hbox = Gtk.HBox()
        vbox.pack_start(hbox, False, False, 0)

        self.entry = Gtk.Entry()
        hbox.add(self.entry)

        self.button = Gtk.Button("Start")
        hbox.pack_start(self.button, False, False, 0)
        self.button.connect("clicked", self.start_stop)
        self.movie_window = Gtk.DrawingArea()
        vbox.add(self.movie_window)
        window.show_all()

        self.player = Gst.ElementFactory.make("playbin", "player")
        vsink = Gst.ElementFactory.make("autovideosink", "vsink")
        self.player.set_property("video-sink", vsink)
        self.pyaudio = None
        asink = self.get_audiosink_bin()
        self.player.set_property("audio-sink", asink)
        bus = self.player.get_bus()
        bus.add_signal_watch()
        bus.enable_sync_message_emission()
        bus.connect("message", self.on_message)
        bus.connect("sync-message::element", self.on_sync_message)

    def get_audiosink_bin(self):
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

    def handoff_cb(self, element, buffer, pad):
        #TODO: Insert watermark

        if self.pyaudio == None:
            self.pyaudio = pyaudio.PyAudio()
            self.stream = self.pyaudio.open(format=self.pyaudio.get_format_from_width(4),
                            channels=1,
                            rate=44100,
                            output=True)
            print("caps: ", pad.get_current_caps().to_string())

        (ret, info) = buffer.map(Gst.MapFlags.READ)
        if ret:
            self.stream.write(info.data)
        print ("output data: ", ret, buffer.pts, info.size)

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

    def start_stop(self, w):
        if self.button.get_label() == "Start":
            filepath = VIDEO_PATH #self.entry.get_text().strip()
            if os.path.isfile(filepath):
                filepath = os.path.realpath(filepath)
                self.button.set_label("Stop")
                self.player.set_property("uri", "file:///" + filepath)
                self.player.set_state(Gst.State.PLAYING)
                print(self.player.get_property("uri"))
                print("START")
        else:
            self.player.set_state(Gst.State.NULL)
            self.button.set_label("Start")
            print("STOP")

    def on_message(self, bus, message):
        #print("on_message")
        t = message.type
        if t == Gst.MessageType.EOS:
            self.player.set_state(Gst.State.NULL)
            self.button.set_label("Start")
        elif t == Gst.MessageType.ERROR:
            self.player.set_state(Gst.State.NULL)
            err, debug = message.parse_error()
            print("Error: %s" % err, debug)
            self.button.set_label("Start")
        # else:
        #     err, debug = message.parse_error()
        #     print("what else: %s" % err, debug)

    def on_sync_message(self, bus, message):
        if message.get_structure().get_name() == 'prepare-window-handle':
            imagesink = message.src
            imagesink.set_property("force-aspect-ratio", True)
            #imagesink.set_window_handle(self.movie_window.get_property('window').get_xid())
            #self._on_video_realize(imagesink)


# call(["gst-launch", "playbin", "uri=\"file:///E:\\\\PycharmProjects\\\\AerialAcousticCommunication\\\\Kalimba.mp3\""])
# time.sleep(1000)

GObject.threads_init()
Gst.init(None)
obj = GTK_Main()
Gtk.main()


# stop stream (4)
# stream.stop_stream()
# stream.close()




####################################################################################

# from pprint import pprint
#
# import gi
# gi.require_version('Gtk', '3.0')
# gi.require_version('Gst', '1.0')
# gi.require_version('GstVideo', '1.0')
#
# from gi.repository import Gtk, Gst
# Gst.init(None)
# Gst.init_check(None)
#
#
# class GstWidget(Gtk.Box):
#     def __init__(self, pipeline):
#         super().__init__()
#         self.connect('realize', self._on_realize)
#         self._bin = Gst.parse_bin_from_description('videotestsrc', True)
#
#     def _on_realize(self, widget):
#         pipeline = Gst.Pipeline()
#         factory = pipeline.get_factory()
#         gtksink = factory.make('gtksink')
#         pipeline.add(gtksink)
#         pipeline.add(self._bin)
#         self._bin.link(gtksink)
#         self.pack_start(gtksink.props.widget, True, True, 0)
#         gtksink.props.widget.show()
#         #pipeline.set_state(Gst.State.PLAYING)
#         pl = factory.make('playbin')
#         #pl = Gst.ElementFactory.make("playbin", "player")
#         asink = Gst.ElementFactory.make("autoaudiosink", "asink")
#         pl.set_property("audio-sink", asink)
#         pl.set_property("uri", "file:///E:\\PycharmProjects\\AerialAcousticCommunication\\Kalimba.mp3")
#         print ("*** ", pl.get_property("uri"))
#         pl.set_state(Gst.State.PLAYING)
#
# window = Gtk.ApplicationWindow()
#
# header_bar = Gtk.HeaderBar()
# header_bar.set_show_close_button(True)
# window.set_titlebar(header_bar)  # Place 2
#
# widget = GstWidget('videotestsrc')
# widget.set_size_request(200, 200)
# window.add(widget)
# window.show_all()
#
# def on_destroy(win):
#     try:
#         Gtk.main_quit()
#     except KeyboardInterrupt:
#         pass
#
# window.connect('destroy', on_destroy)
# Gtk.main()




####################################################################################
# import gi
# gi.require_version('Gtk', '3.0')
# gi.require_version('Gst', '1.0')
#
# from gi.repository import GObject
# from gi.repository import GLib
# from gi.repository import Gtk
# from gi.repository import Gst
#
#
# class PlaybackInterface:
#     def __init__(self):
#         self.playing = False
#
#         # A free example sound track
#         self.uri = "http://cdn02.cdn.gorillavsbear.net/wp-content/uploads/2012/10/GORILLA-VS-BEAR-OCTOBER-2012.mp3"
#
#         # GTK window and widgets
#         self.window = Gtk.Window()
#         self.window.set_size_request(300, 50)
#
#         vbox = Gtk.Box(Gtk.Orientation.HORIZONTAL, 0)
#         vbox.set_margin_top(3)
#         vbox.set_margin_bottom(3)
#         self.window.add(vbox)
#
#         self.playButtonImage = Gtk.Image()
#         self.playButtonImage.set_from_stock("gtk-media-play", Gtk.IconSize.BUTTON)
#         self.playButton = Gtk.Button.new()
#         self.playButton.add(self.playButtonImage)
#         self.playButton.connect("clicked", self.playToggled)
#         Gtk.Box.pack_start(vbox, self.playButton, False, False, 0)
#
#         self.slider = Gtk.HScale()
#         self.slider.set_margin_left(6)
#         self.slider.set_margin_right(6)
#         self.slider.set_draw_value(False)
#         self.slider.set_range(0, 100)
#         self.slider.set_increments(1, 10)
#
#         Gtk.Box.pack_start(vbox, self.slider, True, True, 0)
#
#         self.label = Gtk.Label(label='0:00')
#         self.label.set_margin_left(6)
#         self.label.set_margin_right(6)
#         Gtk.Box.pack_start(vbox, self.label, False, False, 0)
#
#         self.window.show_all()
#
#         # GStreamer Setup
#         Gst.init_check(None)
#         self.IS_GST010 = Gst.version()[0] == 0
#         self.player = Gst.ElementFactory.make("playbin", "player")
#         assert self.player
#         fakesink = Gst.ElementFactory.make("fakesink", "fakesink")
#         self.player.set_property("video-sink", fakesink)
#         bus = self.player.get_bus()
#         # bus.add_signal_watch_full()
#         bus.connect("message", self.on_message)
#         self.player.connect("about-to-finish", self.on_finished)
#
#     def on_message(self, bus, message):
#         t = message.type
#         if t == Gst.Message.EOS:
#             self.player.set_state(Gst.State.NULL)
#             self.playing = False
#         elif t == Gst.Message.ERROR:
#             self.player.set_state(Gst.State.NULL)
#             err, debug = message.parse_error()
#             print
#             "Error: %s" % err, debug
#             self.playing = False
#
#         self.updateButtons()
#
#     def on_finished(self, player):
#         self.playing = False
#         self.slider.set_value(0)
#         self.label.set_text("0:00")
#         self.updateButtons()
#
#     def play(self):
#         self.player.set_property("uri", self.uri)
#         self.player.set_state(Gst.State.PLAYING)
#         GObject.timeout_add(1000, self.updateSlider)
#
#     def stop(self):
#         self.player.set_state(Gst.State.NULL)
#
#     def playToggled(self, w):
#         self.slider.set_value(0)
#         self.label.set_text("0:00")
#
#         if (self.playing == False):
#             self.play()
#         else:
#             self.stop()
#
#         self.playing = not (self.playing)
#         self.updateButtons()
#
#     def updateSlider(self):
#         if (self.playing == False):
#             return False  # cancel timeout
#
#         try:
#             if self.IS_GST010:
#                 nanosecs = self.player.query_position(Gst.Format.TIME)[2]
#                 duration_nanosecs = self.player.query_duration(Gst.Format.TIME)[2]
#             else:
#                 nanosecs = self.player.query_position(Gst.Format.TIME)[1]
#                 duration_nanosecs = self.player.query_duration(Gst.Format.TIME)[1]
#
#             # block seek handler so we don't seek when we set_value()
#             # self.slider.handler_block_by_func(self.on_slider_change)
#
#             duration = float(duration_nanosecs) / Gst.SECOND
#             position = float(nanosecs) / Gst.SECOND
#             self.slider.set_range(0, duration)
#             self.slider.set_value(position)
#             self.label.set_text("%d" % (position / 60) + ":%02d" % (position % 60))
#
#             # self.slider.handler_unblock_by_func(self.on_slider_change)
#
#         except Exception as e:
#             # pipeline must not be ready and does not know position
#             print
#             e
#             pass
#
#         return True
#
#     def updateButtons(self):
#         if (self.playing == False):
#             self.playButtonImage.set_from_stock("gtk-media-play", Gtk.IconSize.BUTTON)
#         else:
#             self.playButtonImage.set_from_stock("gtk-media-stop", Gtk.IconSize.BUTTON)
#
#
# if __name__ == "__main__":
#     PlaybackInterface()
#     Gtk.main()