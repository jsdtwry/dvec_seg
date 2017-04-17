import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

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

# cal means with F-F, M-F, M-M
def cal_1():
    #f = file('init_seg.out').readlines()
    f = file('../know_2_seg/know_twospeaker_seg.out').readlines()

    f_f, m_f, m_m = [], [], []
    for i in f:
        temp = []
        each = i[:-1].split()
        for j in each[1:]:
            temp.append(float(j))
        uu = each[0].split('_')
        if uu[0][0]=='F' and uu[1][0]=='F':
            f_f.append(temp)
        elif uu[0][0]=='M' and uu[1][0]=='M':
            m_m.append(temp)
        else:
            m_f.append(temp)

    re1 = np.mean(f_f, axis=0)
    re2 = np.mean(m_m, axis=0)
    re3 = np.mean(m_f, axis=0)
    #re = np.sum(content, axis=0)
    print len(f_f),
    for i in re1:
        print i,
    print
    print len(m_m),
    for i in re2:
        print i,
    print
    print len(m_f),
    for i in re3:
        print i,
    print

cal_1()

#test()
#plot3dhist()
