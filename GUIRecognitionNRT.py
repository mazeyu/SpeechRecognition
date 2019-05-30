import pyaudio
import wave
import pygame
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from recording import Recorder
import util
import pickle

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 640

pygame.init()
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
pygame.display.set_caption('Speech Recognition')


upImageFilename = 'up.jpg'
downImageFilename = 'All/0/0.jpg'
ys = pickle.load(open('templates.pkl', 'rb'))



state = 'menu'
begin = None
rec = None

def record():
    global state
    state = 'recording'
    global rec
    rec = Recorder()
    global begin
    begin = time.time()
    print("Start recording")
    rec.start()

def stop():
    global state
    state = 'menu'
    global rec
    rec.stop()
    fina = time.time()
    global begin
    t = fina - begin
    print('录音时间为%ds' % t)
    rec.save("test.wav")
    for ii in range(10000000):
        pass
    coeff = util.getCoeff_all('test.wav', 'test.jpg')


    dis = []
    for j in range(len(ys)):
        tmp2 = util.getScore(coeff, ys[j][0])
        # scores[i, ys[j][1]] = min(tmp1, tmp2)
        dis.append((tmp2, ys[j][1]))


    print(sorted(dis))


class Button(object):
    def __init__(self, name, position, size, click_event):
        self.position = position
        self.size = size
        self.name = name
        self.click_event = click_event

    def isOver(self):
        point_x, point_y = pygame.mouse.get_pos()
        x, y = self.position
        w, h = self.size

        in_x = x - w / 2 < point_x < x + w / 2
        in_y = y - h / 2 < point_y < y + h / 2
        return in_x and in_y

    def click(self):
        if self.isOver():
            self.click_event()

    def render(self):
        w, h = self.size
        x, y = self.position

        if self.isOver():
            color = 255, 0, 0
        else:
            color = 0, 255, 0
        width = 0
        text = pygame.font.SysFont('宋体', 50)
        text_fmt = text.render(self.name, 1, color)

        pygame.draw.rect(screen, (122, 122, 122), (x - w / 2, y - h / 2, w, h), width)

        screen.blit(text_fmt, (x - w / 2, y - h / 2))

recordButton = Button("Recording and Recognition", (350, 550), (500, 50), record)
stopButton = Button("Stop Recording", (350, 550), (300, 50), stop)






while True:
    # time.sleep(1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if state == 'menu':
                recordButton.click()
            elif state == 'recording':
                stopButton.click()

    screen.fill((255, 255, 255))
    fig = pygame.image.load('test.jpg').convert_alpha()
    screen.blit(fig, (0, 0))

    if state == 'menu':
        recordButton.render()
    elif state == 'recording':
        stopButton.render()

    pygame.display.update()