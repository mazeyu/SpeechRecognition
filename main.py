import pyaudio
import wave
import pygame
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 640

pygame.init()
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
pygame.display.set_caption('Speech Recognition')



CHUNK = 64000
FORMAT = pyaudio.paInt16
CHANNELS = 1                # 声道数
RATE = 64000          # 采样率
WAVE_OUTPUT_FILENAME = 'test.wav'
p = pyaudio.PyAudio()


sofar = None

frames = []

cnt = 0

def callback(in_data, frame_count, time_info, status):
    global sofar
    global cnt
    print(cnt)
    print(time.time() - start)
    cnt += 1
    # f = np.fromstring(in_data, dtype=np.int16)
    # if sofar is None:
    #     sofar = f
    # else:
    #     sofar = np.concatenate((sofar[-4:], f))
    # if cnt % 100 == 0:
    #     plt.clf()
    #     plt.ylim([-10000, 10000])
    #     plt.plot(sofar)
    #     plt.savefig('test.png')
    #     screen.blit(pygame.image.load('test.png'), (0, 0))
    #     pygame.display.update()
    data = None
    return (data, pyaudio.paContinue)

start = time.time()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

while True:
    # time.sleep(1)
    # wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    # wf.setnchannels(CHANNELS)
    # wf.setsampwidth(p.get_sample_size(FORMAT))
    # wf.setframerate(RATE)
    # wf.writeframes(b''.join(frames))
    # wf.close()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            stream.stop_stream()
            stream.close()
            p.terminate()
            exit()