import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def cal():
    f = file('init_seg.out').readlines()

    content = []
    for i in f:
        temp = []
        each = i[:-1].split()[1:]
        for j in each:
            temp.append(float(j))
        content.append(temp)

    re = np.mean(content, axis=0)\
    #re = np.sum(content, axis=0)
    for i in re:
        print i,


def test():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.random.rand(2, 100) * 4

    hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

    # Construct arrays for the anchor positions of the 16 bars.
    # Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
    # ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
    # with indexing='ij'.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    # Construct arrays with the dimensions for the 16 bars.
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

    plt.show()

def plot3dhist():
    f = file('init_seg.out').readlines()
    FAR, MDR = [], []
    for i in f:
        each = i[:-1].split()[1:]
        FAR.append(float(each[10]))
        MDR.append(float(each[11]))
    FAR = np.array(FAR)
    MDR = np.array(MDR)
    print FAR
    print MDR
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(FAR, MDR, bins=10, range=[[0, 1], [0, 1]])
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    # Construct arrays with the dimensions for the 16 bars.
    dx = 0.04 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    ax.legend()
    ax.set_xlabel('FAR')
    ax.set_ylabel('MDR')
    ax.set_zlabel('Number of Conversations')

    plt.show()

#test()
plot3dhist()