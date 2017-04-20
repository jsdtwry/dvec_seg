import matplotlib.pyplot as plt
import numpy as np


# ff1 = file('result/dvec_0.1_thres.out').readlines()
# ff2 = file('result/bic_0.2_thres.out').readlines()
# ff3 = file('result/bic_0.3_thres.out').readlines()

def read_evl(filename):
    ff = file(filename).readlines()
    roc_x = []
    roc_y = []
    for i in range(100):
        each = ff[i].split(' ')
        if len(each)==3:
            #print each
            roc_x.append(float(each[1]))
            roc_y.append(float(each[2]))
    return roc_x, roc_y


roc_x, roc_y = read_evl('bic_0.1')
plt.plot(roc_x, roc_y, '--', label='bic tolerance 0.1s ')

roc_x, roc_y = read_evl('bic_0.2')
plt.plot(roc_x, roc_y, '-', label='bic tolerance 0.2s')

roc_x, roc_y = read_evl('bic_0.3')
plt.plot(roc_x, roc_y, ':', label='bic tolerance 0.3s')

roc_x, roc_y = read_evl('dvec_0.1')
plt.plot(roc_x, roc_y, 'o-', label='d-vec tolerance 0.1s ')

roc_x, roc_y = read_evl('dvec_0.2')
plt.plot(roc_x, roc_y, '-+', label='d-vec tolerance 0.2s')

roc_x, roc_y = read_evl('dvec_0.3')
plt.plot(roc_x, roc_y, '-*', label='d-vec tolerance 0.3s')

plt.plot([0, 1], [0, 1], '--', color='red')
plt.xlabel("False Alarm(%)")
plt.ylabel("Miss Detection(%)")
plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 11))
plt.grid(True)
plt.legend()
plt.show()
