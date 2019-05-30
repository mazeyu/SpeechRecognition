import pyaudio
import time
import threading
import wave
import pickle
import array

class Recorder():
    def __init__(self, chunk=1024, channels=1, rate=10000):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []

    def start(self):
        threading._start_new_thread(self.__recording, ())

    def __recording(self):
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        data = array.array('h')
        while (self._running):
            data.fromstring(stream.read(self.CHUNK))
            pickle.dump(data, open('data.pkl', 'wb'))
            # self._frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop(self):
        self._running = False


rec = Recorder()
rec.start()
input()
rec.stop()