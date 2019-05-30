import pickle

ys = pickle.load(open('templates.pkl', 'rb'))

import pyaudio
import time
import threading
import wave
from recording import Recorder
import util
import os
import numpy as np


if __name__ == "__main__":

    while True:

        a = int(input('1: 开始识别 2.新建模板'))
        if a == 1:
            rec = Recorder()
            begin = time.time()
            print("Start recording")
            rec.start()
            b = input('请回车停止:')
            print("Stop recording")
            rec.stop()
            fina = time.time()
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
        elif a == 2:
            times = 10
            name = input("请输入模板名，然后立即开始录音，共%d次:" % times)
            try:
                os.mkdir('All/' + name)
            except:
                pass
            for i in range(times):
                rec = Recorder()
                begin = time.time()
                print("Start recording")
                rec.start()
                b = input('请回车停止:')
                print("Stop recording")
                rec.stop()
                fina = time.time()
                t = fina - begin
                print('录音时间为%ds' % t)
                rec.save('All/' + name + '/%d.wav' % i)

            coeff_dict = {}
            for file in os.listdir('All/' + name):
                if file[-4:] != '.wav':
                    continue
                train = 'All/' + name + '/' + file
                coeff = util.getCoeff_all(train, 'All/' + name + '/' + file[:-4] + '.jpg')
                coeff_dict[file] = coeff

            np.random.seed(10)

            allfiles = []
            for file in os.listdir('All/' + name):
                if file[-4:] == '.wav':
                    allfiles.append(file)

            for genNum in range(5):
                np.random.shuffle(allfiles)

                allCoeffs = []
                for j in range(5):
                    coeff = coeff_dict[allfiles[j]]
                    allCoeffs.append(coeff)

                y = util.getModeStable(allCoeffs, 5)
                ys.append((y, name))

            output = open('templates.pkl', 'wb')

            pickle.dump(ys, output)



