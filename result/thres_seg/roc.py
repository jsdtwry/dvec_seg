import matplotlib.pyplot as plt
import numpy as np


# ff1 = file('result/dvec_0.1_thres.out').readlines()
# ff2 = file('result/bic_0.2_thres.out').readlines()
# ff3 = file('result/bic_0.3_thres.out').readlines()

def read_evl(filename):
    ff = file(filename).readlines()
    roc_x = []
    roc_y = []
    for i in ff:
        each = i[:-1].split(' ')
        if len(each)==2:
            print each
            roc_x.append(each[0])
            roc_y.append(each[1])
    return roc_x, roc_y


roc_x, roc_y = read_evl('result_1/bic_0.1.out')
plt.plot(roc_x, roc_y, '--', label='bic tolerance 0.1s ')

roc_x, roc_y = read_evl('result_1/bic_0.2.out')
plt.plot(roc_x, roc_y, '-', label='bic tolerance 0.2s')

roc_x, roc_y = read_evl('result_1/bic_0.3.out')
plt.plot(roc_x, roc_y, ':', label='bic tolerance 0.3s')

roc_x, roc_y = read_evl('result_1/dvec_0.1.out')
plt.plot(roc_x, roc_y, '-,', label='d-vec tolerance 0.1s ')

roc_x, roc_y = read_evl('result_1/dvec_0.2.out')
plt.plot(roc_x, roc_y, '-+', label='d-vec tolerance 0.2s')

roc_x, roc_y = read_evl('result_1/dvec_0.3.out')
plt.plot(roc_x, roc_y, '-*', label='d-vec tolerance 0.3s')

plt.plot([0, 1], [0, 1], '--', color='red')
plt.xlabel("False Alarm(%)")
plt.ylabel("Miss Detection(%)")
plt.xticks(np.linspace(0, 1, 11))
plt.yticks(np.linspace(0, 1, 21))
plt.grid(True)
plt.legend()
plt.show()