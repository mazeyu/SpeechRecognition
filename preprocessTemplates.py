import wave
import numpy as np
import matplotlib.pyplot as plt
import util
import pickle
import os

trick = [7] * 10

# trick[0] = 6
# trick[2] = 6
# trick[8] = 6
# trick[9] = 7
# trick[7] = 4


ys = []

set = ['%d' % i for i in range(10)]
coeff_dict = {}


for i in range(len(set)):
    for file in os.listdir('All/' + set[i]):
        if file[-4:] != '.wav':
            continue
        train = 'All/' + set[i] + '/' + file
        coeff = util.getCoeff_all(train, 'All/' + set[i] + '/' + file[:-4] + '.jpg')

        coeff_dict[(set[i], file)] = coeff


c = ['b', 'k', 'w', 'r', 'g']

for i in range(len(set)):
    plt.figure(10000 + i)
    for file in os.listdir('All/' + set[i]):
        if file[-4:] != '.wav':
            continue
        coeff = coeff_dict[(set[i], file)]
        np.random.shuffle(c)
        for j in range(len(coeff)):
            plt.scatter(coeff[j][2], coeff[j][1], c=c)
    plt.savefig('traj %d.jpg' % i)


np.random.seed(10)


for i in range(len(set)):
    allfiles = []
    for file in os.listdir('All/' + set[i]):
        if file[-4:] == '.wav':
            allfiles.append(file)

    for genNum in range(5):
        np.random.shuffle(allfiles)

        allCoeffs = []
        for j in range(5):

            coeff = coeff_dict[(set[i], allfiles[j])]
            allCoeffs.append(coeff)

        y = util.getModeStable(allCoeffs, trick[i])
        ys.append((y, set[i]))


output = open('templates.pkl', 'wb')


pickle.dump(ys, output)