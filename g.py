import gi
import sys
import pygame
import threading
import time
pygame.init()
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

class ButtonWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Moves")
        # self.set_border_width(10)
        self.set_border_width(20)
        self.set_default_size(300, 100)

        hbox = Gtk.Box(orientation = Gtk.Orientation.HORIZONTAL,spacing=6)
        vbox = Gtk.Box(orientation = Gtk.Orientation.HORIZONTAL,spacing=6)
        mainbox = Gtk.Box(spacing=12)
        mainbox.pack_start(hbox,True,True,0)
        mainbox.pack_start(vbox,True,True,0)
        mainbox.set_homogeneous(False)
        vbox.set_homogeneous(False)
        hbox.set_homogeneous(False)
        self.add(mainbox)

        # self.main_box = Gtk.Box(
        #     orientation=Gtk.Orientation.VERTICAL, spacing=6)
        # self.main_box.set_size_request(350, 700)

        button = Gtk.Button.new_with_label("Play")
        button.connect("clicked", self.on_click_me_clicked)
        hbox.pack_start(button, True, True, 0)
        self.label = Gtk.Label("Moves appear here")
        self.label.set_justify(Gtk.Justification.RIGHT)
        vbox.pack_start(self.label, True, True, 0)
        self.song_thread = myT(lock,self.label)

    def on_click_me_clicked(self, button):
        print(sys.argv)
        self.song_thread.start()
        # f = open(sys.argv[2],'r')
        # g = f.readlines()
        # for line in g:
        #     self.label.set_text(line)
        #     time.sleep(1)
        # f.close()
    def on_quit(self, action, param):
        myT.cancel()
        self.quit()
        Gtk.main_quit()
def play_song(label):
    song = pygame.mixer.Sound(sys.argv[1])
    clock = pygame.time.Clock()
    counter = 0
    song.play()
    f = open(sys.argv[2],'r')
    g = f.readlines()
    while counter < len(g):
        lock.acquire()
        # print(g[counter])
        label.set_text(g[counter])
        lock.release()
        counter += 1
        time.sleep(1)
        # clock.tick(1)
    pygame.quit()

class myT(threading.Thread):
    def __init__(self,lock,label):
        threading.Thread.__init__(self)
        self.lock = lock
        self.label = label
    def run(self):
        play_song(self.label)


lock = threading.Lock()
win = ButtonWindow()

win.connect("delete-event", Gtk.main_quit)
win.show_all()
Gtk.main()
