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
    #block_rep.append(np.mean(feat_vad[start_index:], axis=0))
    #block_tag.append([start_index, len(feat_vad)])
    #print start_index, len(feat_vad)
    return block_rep, block_tag

'''
speaker clustering using k-means, default K=2
'''
def spk_k_means_cluster(det_index, feat_vad, feat_time, type):
    det_index.append(len(feat_vad))
    #print det_index
    spk0_rep, spk1_rep = [], []
    cluster_change_point, cluster_result = [], []
    last_index = 0
    block_rep, block_tag = gen_block_representation(det_index, feat_vad, feat_time)
    y_pred = KMeans(n_clusters=2).fit_predict(block_rep)
    for i, k in enumerate(y_pred):
        #print last_index, block_tag[i]
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
    if k==0:
        end = 1
    else:
        end = 0
    cluster_result.append([[last_index, block_tag[i][1]-1], end])
    if type=='mean': 
        spk0_model = np.mean(spk0_rep, axis=0)
        spk1_model = np.mean(spk1_rep, axis=0)

        return cluster_change_point, cluster_result, [spk0_model, spk1_model]
    else:
        clf = svm.SVC()
        clf.fit(block_rep, y_pred)
        return cluster_change_point, cluster_result, clf

'''
generate dvector speaker models
'''
def get_spk_model(feat, utt_name):
    reflist = file('../thu_ev_tag/'+utt_name+'.txt').readlines()
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
    
 
'''
speaker clustering and combination with given speaker models
'''
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

'''speaker clustering and combineation with only one speaker model'''
def spk_reseg_with_one_model(det_index, feat_vad, feat_time, spk_model):
    det_index.append(len(feat_vad))
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
    if k==0:
        end = 1
    else:
        end = 0
    cluster_result.append([[last_index, block_tag[i][1]-1], end])
    
    spk0_model = np.mean(spk0_rep, axis=0)
    spk1_model = np.mean(spk1_rep, axis=0)

    cluster_change_point, cluster_result = [], []
    cos_m0 = np.dot(spk_model,spk0_model)/(np.linalg.norm(spk_model)*np.linalg.norm(spk0_model))
    cos_m1 = np.dot(spk_model,spk1_model)/(np.linalg.norm(spk_model)*np.linalg.norm(spk1_model))
    if cos_m0 < cos_m1:
        spk0_model = spk_model
    else:
        spk1_model = spk_model
    
    last_index = 0
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
    if k==0:
        end = 1
    else:
        end = 0
    cluster_result.append([[last_index, block_tag[i][1]-1], end])
    return cluster_change_point, cluster_result



