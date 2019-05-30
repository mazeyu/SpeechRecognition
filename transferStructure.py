import os, shutil

for i in range(12):
    for j in range(10):
        src = 'b%d/%d.wav' % (i, j)
        dst = 'All/%d/%d.wav' % (j, i)
        shutil.copy(src, dst)
for i in range(7):
    for j in range(10):
        src = 't%d/%d.wav' % (i, j)
        dst = 'All/%d/%d.wav' % (j, 12 + i)
        shutil.copy(src, dst)
