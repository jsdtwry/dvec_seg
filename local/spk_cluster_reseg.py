import numpy as np
from sklearn.cluster import KMeans
from sklearn import svm
from initial_seg import *

def gen_block_representation(det_index, feat_vad, feat_time):
    start_index = 0
    block_rep = []
    block_tag = []
    for i in det_index:
        block_rep.append(np.mean(feat_vad[start_index:i], axis=0))
        block_tag.append([start_index, i])
        start_index = i
    return block_rep, block_tag

'''
speaker clustering using k-means, default K=2
'''
def spk_k_means_cluster(det_index, feat_vad, feat_time, type):
    spk0_rep, spk1_rep = [], []
    cluster_change_point, cluster_result = [], []
    last_index = 0
    block_rep, block_tag = gen_block_representation(det_index, feat_vad, feat_time)
    y_pred = KMeans(n_clusters=2).fit_predict(block_rep)
    for i, k in enumerate(y_pred):
        if k==0:
            for j in range(block_tag[i][0], block_tag[i][1]):
                spk0_rep.append(feat_vad[j])
        else:
            for j in range(block_tag[i][0], block_tag[i][1]):
                spk1_rep.append(feat_vad[j])
        if i==0:
            continue
        if k!=y_pred[i-1]:
            cluster_change_point.append(block_tag[i][0])
            cluster_result.append([[last_index, block_tag[i][0]], k])
            last_index = block_tag[i][0]
    if type=='mean': 
        spk0_model = np.mean(spk0_rep, axis=0)
        spk1_model = np.mean(spk1_rep, axis=0)

        return cluster_change_point, cluster_result, [spk0_model, spk1_model]
    else:
        clf = svm.SVC()
        clf.fit(block_rep, y_pred)
        return cluster_change_point, cluster_result, clf


def get_spk_model(feat, utt_name):
    reflist = file('../170309_thu_ev/thu_ev_tag/'+utt_name+'.txt').readlines()
    spk1 = utt_name.split('_')[0][:-3]
    spk2 = utt_name.split('_')[1][:-3]
    #print spk1, spk2
    
    spk_dict = {spk1:[],spk2:[]}
    for i in reflist:
        each = i.split()
        if each[2]=='SIL' or each[2]=='OVERLAP':
            continue
        spk_dict[each[2]].append([time2frame(float(each[0]), mfcc_shift),time2frame(float(each[1]), mfcc_shift)])
    spk1_data, spk2_data = [], []
    
    for i in spk_dict[spk1]:
        spk1_data += feat[i[0]:i[1]].tolist()
    for i in spk_dict[spk2]:
        spk2_data += feat[i[0]:i[1]].tolist()
    spk0_rep = np.array(spk1_data)
    spk1_rep = np.array(spk2_data)
    spk0_model = np.mean(spk0_rep, axis=0)
    spk1_model = np.mean(spk1_rep, axis=0)
    return spk0_model, spk1_model
    
 

def spk_reseg_with_models(det_index, feat_vad, feat_time, spk_model):
    cluster_change_point, cluster_result = [], []
    spk0_model, spk1_model = spk_model
    last_index = 0
    block_rep, block_tag = gen_block_representation(det_index, feat_vad, feat_time)
    y_pred = []
    for i in block_rep:
        cos_0 = np.dot(i,spk0_model)/(np.linalg.norm(i)*np.linalg.norm(spk0_model))
        cos_1 = np.dot(i,spk1_model)/(np.linalg.norm(i)*np.linalg.norm(spk1_model))
        if cos_0 < cos_1:
            y_pred.append(0)
        else:
            y_pred.append(1)
    for i, k in enumerate(y_pred):
        if i==0:
            continue
        if k!=y_pred[i-1]:
            cluster_change_point.append(block_tag[i][0])
            cluster_result.append([[last_index, block_tag[i][0]], k])
            last_index = block_tag[i][0]
    return cluster_change_point, cluster_result


'''
#scores, times, det_time, det_index = initial_segmentation('data/F001HJN_F002VAN_001/mfcc_feats.ark', 'data/F001HJN_F002VAN_001/fbank_vad.ark', 20, 1, 0.1, 'bic')

mfcc_file = 'data/F001HJN_F002VAN_001/mfcc_feats.ark' # 20 dim
dvector_file = 'data/F001HJN_F002VAN_001/dvector.ark' # 400 dim
vad_file = 'data/F001HJN_F002VAN_001/fbank_vad.ark'

utt_lable, content = readfeatfromkaldi(mfcc_file, 20)
print utt_lable
print len(content)
print content[0]

vad_utt_label, vad_content = readvadfromkaldi(vad_file)

print vad_utt_label
print len(vad_content)
print vad_content[0]

feat_vad, feat_time = gen_feat_vad(content,  vad_content)
#scores, times, det_time, det_index = initial_seg(feat_vad, feat_time, 0.1, 0.01, 'dvec')
scores, times, det_time, det_index = initial_seg(feat_vad, feat_time, 1, 0.1, 'bic', lamda=1.0)

change_point, segment_result, spk_model = spk_k_means_cluster(det_index, feat_vad, feat_time, 'svm')
print len(change_point), len(segment_result), spk_model
# print change_point
# print segment_result
print [frame2time(feat_time[i], mfcc_shift) for i in change_point]
print [[[frame2time(feat_time[i[0][0]], mfcc_shift), frame2time(feat_time[i[0][1]], mfcc_shift)], i[1]] for i in segment_result]
'''



