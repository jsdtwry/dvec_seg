import sys
import numpy as np
from sklearn.decomposition import PCA

def readfeatfromkaldi(filename,dim):
    fb = file(filename).readlines()
    utt_lable = []
    content = []
    utt_content = []
    for i in fb:
        if i[-2]=='[':
            utt_lable.append(i.split('  ')[0])
            continue
        each = i.split(' ')
        frame_content = []
        for j in each[2:dim+2]:
            frame_content.append(float(j))
        utt_content.append(frame_content)
        if each[-1]==']\n':
            content.append(utt_content)
            utt_content = []

    return utt_lable, content

def readvadfromkaldi(filename):
    vb = file(filename).readlines()
    utt_lable = []
    content = []
    utt_content = []
    for i in vb:
        vad = i.split(' ')
        utt_lable.append(vad[0])
        content.append(vad[3:-1])
    return utt_lable, content

def save_feat_pca(utt_name):
    featfile = 'data/'+utt_name+'/dvector.ark'
    vadfile = 'data/'+utt_name+'/fbank_vad.ark'
    utt_lable, content = readfeatfromkaldi(featfile, 400)
    vad_utt_label, vad_content = readvadfromkaldi(vadfile)
    feat = content[0]
    vad = vad_content[0]

    feat_vad= []
    for index, i in enumerate(feat):
        if(vad[index]=='1'):
            feat_vad.append(i)

    pca = PCA(n_components=40)
    feat = pca.fit(feat_vad).transform(feat)
    np.save("feat_pca/"+utt_name+".npy",feat)

#save_feat_pca('fe_03_05901')


flist =  file('lst/thu_ev.lst').readlines()
for i in flist:
        each = i[:-1].split(' ')[0]
        print each
	save_feat_pca(each)
