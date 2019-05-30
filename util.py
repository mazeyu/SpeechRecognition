import numpy as np
import matplotlib.pyplot as plt
import wave

def shortTimeEnergy(data, fs):
    win = int(fs * 0.01)
    ker = [1 / win] * win
    return np.convolve(data ** 2, ker, mode='same')


def zeroRate(data, fs):
    win = int(fs * 0.01)
    ker = [1 / win] * win
    tmp = np.sign(data)
    tmp[:-1] -= tmp[1:]
    return np.convolve(np.abs(tmp) / 2, ker, mode='same')


def plotStd(data):
    plt.plot(data / np.max(np.abs(data)))

def getStEd(data, th1, th2, fs):
    st, ed = 0, 0
    state = 0
    for i in range(len(data)):
        if state == 0:
            if data[i] > th2:
                st = i
                state = 1
        if state == 1:
            if data[i] < th1:
                ed = i
                if ed - st > 0.1 * fs:
                    state = 2
                else:
                    state = 0
        if state == 2:
            if data[i] > th2:
                if i - ed < 0.02 * fs:
                    state = 1
    return st, ed

def divide(data, win, step):
    ret = []
    preAcc = data.astype(np.float)
    preAcc[1:] -= preAcc[:-1] * 0.96
    for i in range(0, len(data) - win, step):
        ret.append(preAcc[i: i + win] * np.hamming(win))
    return np.array(ret)



def lpc(R):
    N = R.shape[0] - 1
    a = np.zeros((N + 1, N + 1))
    B2 = np.zeros(N + 1)
    K = np.zeros(N + 1)

    # initial condition
    B2[0] = R[0]
    a[0][0] = 1

    # recursion
    for i in range(1, N + 1):
        K[i] = -a[i - 1][i - 1:: -1].dot(R[1: i + 1]) / B2[i - 1]
        B2[i] = (1 - K[i] ** 2) * B2[i - 1]
        a[i][i] = K[i]
        a[i][1: i] = a[i - 1][1: i] + K[i] * a[i - 1][i - 1: 0: -1]

        a[i][0] = 1
    # output
    # for i in range(1, N + 1):
    #     print('alpha %d: ' % i)
    #     print(a[i][1:])
    # print('K: ')
    # print(K[1:])

    return a[N]

def getMode(data, n):
    N, P = data.shape
    t = np.zeros(n + 1, dtype=np.int)
    y = np.zeros((n, P))
    for i in range(n):
        t[i] = int(i * N / n)
    t[n] = N
    pret = t.copy()
    for i in range(n):
        tmp = data[t[i]: t[i + 1]]
        y[i] = np.mean(tmp, axis=0)

    pre = np.zeros((N, n), dtype=np.int)
    d = np.zeros((N, n))
    d[0, 0] = np.sum((data[0] - y[0]) ** 2)
    for i in range(1, n):
        d[0, i] = np.inf
    while True:
        for i in range(1, N):
            for j in range(n):
                d[i, j] = np.sum((data[i] - y[j]) ** 2)
                if j == 0:
                    d[i, j] += d[i - 1, j]
                    pre[i, j] = j
                else:
                    if d[i - 1, j] < d[i - 1, j - 1]:
                        d[i, j] += d[i - 1, j]
                        pre[i, j] = j
                    else:
                        d[i, j] += d[i - 1, j - 1]
                        pre[i, j] = j - 1
        cur = N - 1
        cury = n - 1
        t = []
        while cur != 0:
            if pre[cur, cury] != cury:
                t.append(cur)
            cury = pre[cur, cury]
            cur -= 1
        t.append(0)
        t = t[::-1]
        t.append(N)
        t = np.array(t)
        if (t == pret).all():
            # print(d[N - 1, n - 1])
            # print(t)

            break
        pret = t.copy()
        for i in range(n):
            tmp = data[t[i]: t[i + 1]]
            y[i] = np.mean(tmp, axis=0)

    return t, y.copy(), d[N - 1, n - 1]

def dis(x, y):
    tmp = np.abs(x - y)
    # for i in range(len(tmp)):
    #     tmp[i] = max(tmp[i], 0)
    return np.sum(tmp ** 2)

def getScore(data, y):
    N, P = data.shape
    n = y.shape[0]
    d = np.zeros((2, n))
    d[0, 0] = dis(data[0], y[0])
    for i in range(1, n):
        d[0, i] = np.inf

    for i in range(1, N):
        for j in range(n):
            d[i % 2, j] = dis(data[i], y[j])
            if j == 0:
                d[i % 2, j] += d[(i - 1) % 2, j]
            else:
                d[i % 2, j] += np.min(d[(i - 1) % 2, :j + 1])
    return d[(N - 1) % 2, n - 1]


def calLog(a):
    N = len(a)
    b = np.zeros(N)
    b[0] = 1
    for i in range(1, N):
        for j in range(i):
            b[i] -= b[j] * a[i - j]
    c = np.zeros(N)
    for i in range(N - 1):
        for j in range(N - i):
            c[i + j] += b[j] * a[i + 1] * (i + 1)
    d = np.zeros(N)
    for i in range(1, N):
        d[i] = - c[i - 1] / i
    return d[1:]


def getModeStable(datas, n):
    B = len(datas)
    _, P = datas[0].shape
    t = np.zeros((B, n + 1), dtype=np.int)
    y = np.zeros((n, P))
    for b in range(B):
        N = datas[b].shape[0]
        for i in range(n):
            t[b, i] = int(i * N / n)
        t[b, n] = N

    pret = t.copy()

    for i in range(n):
        tmp = []
        for b in range(B):
            tmp.append(datas[b][t[b, i]: t[b, i + 1]])
        y[i] = np.mean(np.concatenate(tmp), axis=0)
    while True:
        for b in range(B):
            N = datas[b].shape[0]
            data = datas[b]
            pre = np.zeros((N, n), dtype=np.int)
            d = np.zeros((N, n))
            d[0, 0] = np.sum((data[0] - y[0]) ** 2)
            for i in range(1, n):
                d[0, i] = np.inf

            for i in range(1, N):
                for j in range(n):
                    d[i, j] = np.sum((data[i] - y[j]) ** 2)
                    if j == 0:
                        d[i, j] += d[i - 1, j]
                        pre[i, j] = j
                    else:
                        if d[i - 1, j] < d[i - 1, j - 1]:
                            d[i, j] += d[i - 1, j]
                            pre[i, j] = j
                        else:
                            d[i, j] += d[i - 1, j - 1]
                            pre[i, j] = j - 1
            cur = N - 1
            cury = n - 1
            t_ = []
            while cur != 0:
                if pre[cur, cury] != cury:
                    t_.append(cur)
                cury = pre[cur, cury]
                cur -= 1
            t_.append(0)
            t_ = t_[::-1]
            t_.append(N)
            t[b] = np.array(t_)

        if (t == pret).all():
            break

        pret = t.copy()
        for i in range(n):
            tmp = []
            for b in range(B):
                tmp.append(datas[b][t[b, i]: t[b, i + 1]])
            y[i] = np.mean(np.concatenate(tmp), axis=0)

    return y.copy()


def getStEd_all(wave_data, framerate, saveFile):
    energy = shortTimeEnergy(wave_data, framerate)
    rate = zeroRate(wave_data, framerate)

    if not saveFile is None:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(rate)

        plt.subplot(3, 1, 2)
        plt.plot(energy)

        plt.subplot(3, 1, 3)
        plotStd(wave_data)
        plotStd(energy)
        plotStd(rate)

    st, ed = getStEd(energy, 10000, 70000, framerate)
    st1 = st - int(0.3 * framerate)
    while rate[st1] < 0.2 and st1 < st:
        st1 += 1

    if st1 < st:
        while rate[st1] > 0.1 and st1 > 0:
            st1 -= 1
        st = st1

    if not saveFile is None:
        plt.plot((st, st + 1), (-1, 1))
        plt.plot((ed, ed + 1), (-1, 1))

        plt.savefig(saveFile)
    # plt.show()
    # if not verbose:
    #     ed -= 2000
    return st, ed


def getCoeff(frame):
    # dft
    x = np.fft.fft(frame)
    lnx = np.log(x)
    # rectify the angle
    for i in range(1, lnx.shape[0]):
        if np.imag(lnx[i]) > np.imag(lnx[i - 1]) + np.pi:
            lnx[i:] -= 2 * np.pi * np.complex(0, 1)
        elif np.imag(lnx[i]) < np.imag(lnx[i - 1]) - np.pi:
            lnx[i:] += 2 * np.pi * np.complex(0, 1)
    # idft
    xhat = np.fft.ifft(lnx)
    # plt.figure()
    # plt.plot(xhat[1: 21])
    # plt.show()
    return xhat[1: 21]

def getCoeffLPC(frame):
    P = 18
    N = len(frame)
    R = np.correlate(frame, frame, 'full')[N - 1: N + P]
    a = lpc(R)

    cep = calLog(a)
    dif = cep[:-1] - cep[1:]
    ret = np.concatenate((cep, dif[:10]))
    return cep




def getCoeff_all(filename, saveFile):
    f = wave.open(filename, "rb")

    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()

    wave_data = np.fromstring(str_data, dtype=np.short).astype(np.int)

    st, ed = getStEd_all(wave_data, framerate, saveFile)

    if ed < st: ed = len(wave_data)
    data = divide(wave_data[st: ed], int(0.04 * framerate), int(0.02 * framerate))

    print(data.shape)
    # plt.figure()

    coeff = []
    for j in range(len(data)):
        a = getCoeffLPC(data[j])
        coeff.append(a)
        # plt.figure()
        # plt.plot(a)
        # plt.savefig('%d.jpg' % i)
    # plt.show()

    coeff = np.array(coeff)
    return coeff
