import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.decomposition import PCA

mfcc_len = 0.10
mfcc_shift = 0.01
ft = 0.3
spk_model_type = 'mean'

def time2frame(time, mfcc_shift):
    return int(time*1/mfcc_shift)

def frame2time(frame, mfcc_shift):
    return float(frame/(1/mfcc_shift))

def inline(point, ref, ft):
    for i in ref:
        if point >= i[0]-ft and point <= i[1]+ft:
            return True
    return False

def inpoint(seg, det, ft):
    for i in det:
        if float(i) >= seg[0]-ft and float(i) <= seg[1]+ft:
            return True
    return False

def inpoint_t(point, list_tag, tag):
    for i in list_tag:
        if i[1]==tag and point>=i[0][0] and point <= i[0][1]:
            return True
    return False

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

def gen_feat_vad(feat, vad):
    new_feat = []
    feat_time = []
    for index, i in enumerate(zip(feat, vad)):
        #print i[1], index
        if i[1]!='0':
            new_feat.append(i[0])
            feat_time.append(index)
    # for i in new_feat:
    #     print i
    return new_feat, feat_time

def read_ref(filename):
    ref = []
    utt = file(filename).readlines()
    utt_t = utt[3:]
    for index, each in enumerate(utt_t):
        if each=='\n':
            continue
        line = each.split(' ')
        if line[2][-1]==':':
            start = float(line[0])
            end = float(line[1])
            spk = line[2][:-1]
        ref.append([[start, end], int(spk=='B')])
    return ref

'''
1. re-segmentation after clustering on ground truth;
2. re-segmentation after clustering on ground truth and initial segmentation
'''
def get_spk_model_ground_truth(ref, feat, type):
    spk0_rep = []
    spk1_rep = []
    spk_rep = []
    y_pred = []
    if type=='mean':
        for i in ref:
            if i[1]==0:
                #spk0_rep = spk0_rep + feat[int(i[0][0]*100):int(i[0][1]*100)]
                spk0_rep.append(np.mean(feat[int(i[0][0]*100):int(i[0][1]*100)], axis=0))
            else:
                #spk1_rep = spk1_rep + feat[int(i[0][0]*100):int(i[0][1]*100)]
                spk1_rep.append(np.mean(feat[int(i[0][0]*100):int(i[0][1]*100)], axis=0))
        spk0_model = np.mean(spk0_rep, axis=0)
        spk1_model = np.mean(spk1_rep, axis=0)
        return [spk0_model, spk1_model]
    else:
        for i in ref:
            spk_rep.append(np.mean(feat[int(i[0][0]*100):int(i[0][1]*100)], axis=0))
            y_pred.append(i[1])
        clf = svm.SVC()
        clf.fit(spk_rep, y_pred)
        return clf

def cpt_seg(filename):
    utt = file(filename).readlines()
    cpt_seg = []
    utt_t = utt[3:]
    for index, each in enumerate(utt_t):
        if each=='\n':
            continue
        line = each.split(' ')
        if line[2][-1]==':':
            start = float(line[0])
            end = float(line[1])
            spk = line[2]

            if index==0:
                    continue
            if spk!=utt_t[index-2].split(' ')[2]:
                cpt_seg.append(sorted([float(utt_t[index-2].split(' ')[1]), start]))
    return cpt_seg

def dvec_cos(left, right):
    leftvec = np.mean(np.array(left), axis=0)
    rightvec = np.mean(np.array(right), axis=0)
    cos = np.dot(leftvec,rightvec)/(np.linalg.norm(leftvec)*np.linalg.norm(rightvec))
    return cos

def fix_slid(utt_vad, utt_vad_index, win_len, win_shift):
    times = []
    scores = []
    f_len = len(utt_vad)
    startpoint = 0
    while startpoint + time2frame(win_len*2, mfcc_shift) < f_len:
        mid = startpoint + time2frame(win_len, mfcc_shift)
        left_features = utt_vad[mid-time2frame(win_len, mfcc_shift):mid]
        right_features = utt_vad[mid:mid+time2frame(win_len, mfcc_shift)]
        full_features = utt_vad[mid-time2frame(win_len, mfcc_shift):mid+time2frame(win_len, mfcc_shift)]
        scores.append(dvec_cos(left_features, right_features))
        times.append(frame2time(utt_vad_index[mid], mfcc_shift))
        startpoint += time2frame(win_shift, mfcc_shift)
    return times, scores

def fix_slid_det_dvec_nothreshold(times, times_index, scores):
    det, det_index = [], []
    for index, each in enumerate(scores):
            if index == 0 or index == len(scores)-1:
                    continue
            if each < scores[index-1] and each < scores[index+1]:
                    det.append(times[index])
                    det_index.append(times_index[index])
    return det, det_index

'''Init segmentation with no threshold return the list of change point and change point ref'''
'''[1.92, 2.16, 2.65, 2.78, 2.88, 3.14]'''
def initial_seg(feat_vad, feat_time):
    win_len, win_shift = 0.1, 0.01
    scores, times, times_index = [], [], []
    f_len = len(feat_vad)
    startpoint = 0
    while startpoint + time2frame(win_len*2, mfcc_shift) < f_len:
        mid = startpoint + time2frame(win_len, mfcc_shift)
        #print startpoint, mid, mid+time2frame(win_len, mfcc_shift)
        left_features = feat_vad[mid-time2frame(win_len, mfcc_shift):mid]
        right_features = feat_vad[mid:mid+time2frame(win_len, mfcc_shift)]
        full_features = feat_vad[mid-time2frame(win_len, mfcc_shift):mid+time2frame(win_len, mfcc_shift)]
        scores.append(dvec_cos(left_features, right_features))
        times.append(frame2time(feat_time[mid], mfcc_shift))
        times_index.append(mid)
        startpoint += time2frame(win_shift, mfcc_shift)
    #print len(times), len(scores)
    det_time, det_index = fix_slid_det_dvec_nothreshold(times, times_index, scores)
    #print det_tmp
    return det_time, det_index

def gen_block_representation(det_index, feat_vad, feat_time):
    start_index = 0
    block_rep = []
    block_tag = []
    for i in det_index:
        block_rep.append(np.mean(feat_vad[start_index:i], axis=0))
        block_tag.append([start_index, i])
        start_index = i
    return block_rep, block_tag

def initial_cluster(det_index, feat_vad, feat_time, type):
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

def re_seg_block(det_index, feat_vad, feat_time, spk_model, type):
    spk0_rep, spk1_rep = [], []
    cluster_change_point, cluster_result = [], []
    last_index = 0
    y_pred = []
    block_rep, block_tag = gen_block_representation(det_index, feat_vad, feat_time)
    if type=='mean':
        for i, k in enumerate(block_rep):
            cos_0 = np.dot(k,spk_model[0])/(np.linalg.norm(k)*np.linalg.norm(spk_model[0]))
            cos_1 = np.dot(k,spk_model[1])/(np.linalg.norm(k)*np.linalg.norm(spk_model[1]))
            if cos_0 < cos_1:
                y_pred.append(0)
            else:
                y_pred.append(1)
    else:
        y_pred = spk_model.predict(block_rep)

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

def re_seg_frame(feat_vad, feat_time, spk_model, type):
    cluster_change_point, cluster_result= [], []
    y_pred = []
    spk0_rep, spk1_rep = [], []
    count = 0

    if type=='mean':
        for i, k in enumerate(feat_vad):
            cos_0 = np.dot(k,spk_model[0])/(np.linalg.norm(k)*np.linalg.norm(spk_model[0]))
            cos_1 = np.dot(k,spk_model[1])/(np.linalg.norm(k)*np.linalg.norm(spk_model[1]))

            if cos_0 < cos_1:
                y_pred.append(0)
            else:
                y_pred.append(1)
    else:
        y_pred = spk_model.predict(feat_vad)

    for i, k in enumerate(y_pred):
        if k==0:
            spk0_rep.append(feat_vad[i])
        else:
            spk1_rep.append(feat_vad[i])
        if i==0:
            continue
        if k==y_pred[i-1]:
            count+=1
        if k!=y_pred[i-1]:
            cluster_change_point.append(i)
            count=0

    if type=='mean': 
        spk0_model = np.mean(spk0_rep, axis=0)
        spk1_model = np.mean(spk1_rep, axis=0)

        return cluster_change_point, [spk0_model, spk1_model]
    else:
        clf = svm.SVC()
        clf.fit(feat_vad, y_pred)
        return cluster_change_point, clf

'''
0: a change point in the block
1: no change points in the block
'''
def conf_judge_ground_truth(cluster_result, feat_time, ref):
    conf = []
    for i in cluster_result:
        for index, j in enumerate(ref):
            if j[0]-frame2time(feat_time[i[0][0]], mfcc_shift) > ft and frame2time(feat_time[i[0][1]], mfcc_shift)-j[1] > ft:
                conf.append(0)
                break
            if index+1==len(ref):
                conf.append(1)
    return conf

def frame_reseg_low_purity_block(cluster_result, conf, feat_vad, feat_time, spk_model, type):
    reseg_result = []
    for i in zip(cluster_result, conf):
        if i[1] == 1:
            reseg_result.append(i[0])
        else:
            if type=='mean':
                y_pred = []
                for k in feat_vad[int(i[0][0][0]): int(i[0][0][1])]:
                    cos_0 = np.dot(k,spk_model[0])/(np.linalg.norm(k)*np.linalg.norm(spk_model[0]))
                    cos_1 = np.dot(k,spk_model[1])/(np.linalg.norm(k)*np.linalg.norm(spk_model[1]))
                    if cos_0 < cos_1:
                        y_pred.append(0)
                    else:
                        y_pred.append(1)
            else:
                y_pred = spk_model.predict(feat_vad[int(i[0][0][0]): int(i[0][0][1])])
            count = 0
            last_time = i[0][0][0]
            # print i[0][0],
            # print y_pred
            for ii, k in enumerate(y_pred):
                if ii==0:
                    continue
                if k==y_pred[ii-1]:
                    count+=1
                if k!=y_pred[ii-1]:
                #if k!=y_pred[ii-1]:
                    reseg_result.append([[last_time, i[0][0][0]+ii], y_pred[ii-1]])
                    last_time = i[0][0][0] + ii
                    count=0
            reseg_result.append([[last_time, i[0][0][1]], y_pred[len(y_pred)-1]])

    change_point = []

    for index, i in enumerate(reseg_result):
        if index==0:
            last_time = i[0][0]
            continue
        if i[1]!=reseg_result[index-1][1]:
            change_point.append(i[0][0])
            last_time = i[0][0]

    return re_seg_block(change_point, feat_vad, feat_time, spk_model, type)


def block_reseg_low_purity_block(det_index, cluster_result, conf, feat_vad, feat_time, spk_model, type):
    reseg_result = []
    block_rep, block_tag = gen_block_representation(det_index, feat_vad, feat_time)
    for i in zip(cluster_result, conf):
        if i[1] == 1:
            reseg_result.append(i[0])
        else:
            #print i[0],'========'
            for index, ii in enumerate(block_tag):
                if ii[0]==i[0][0][0]:
                    start_index = index
                if ii[1]==i[0][0][1]:
                    end_index = index
            reseg_content = block_rep[start_index: end_index+1]
            reseg_content_tag = block_tag[start_index: end_index+1]
            
            if type=='mean':
                y_pred = []
                for k in reseg_content:
                    cos_0 = np.dot(k,spk_model[0])/(np.linalg.norm(k)*np.linalg.norm(spk_model[0]))
                    cos_1 = np.dot(k,spk_model[1])/(np.linalg.norm(k)*np.linalg.norm(spk_model[1]))
                    if cos_0 < cos_1:
                        y_pred.append(0)
                    else:
                        y_pred.append(1)
            else:
                y_pred = spk_model.predict(reseg_content)

            for i in zip(reseg_content_tag, y_pred):
                reseg_result.append([i[0], i[1]])

    change_point = []

    for index, i in enumerate(reseg_result):
        if index==0:
            last_time = i[0][0]
            continue
        if i[1]!=reseg_result[index-1][1]:
            change_point.append(i[0][0])
            last_time = i[0][0]

    return re_seg_block(change_point, feat_vad, feat_time, spk_model, type)


def seg_evlaution(det_tmp, ref_f):
    false_alarm = 0
    miss_det = 0
    total = 0
    det = []
    for i in det_tmp:
        if i > ref_f[0][0] and i < ref_f[-1][-1]:
            det.append(i)

    for i in det:
        if not inline(float(i), ref_f, ft):
            false_alarm += 1

    for i in ref_f:
        if not inpoint(i, det, ft):
            miss_det += 1
    #print det
    total = len(ref_f)
    FAR = float(false_alarm)/(total+false_alarm)
    #FAR = float(false_alarm)/(len(det))
    MDR = float(miss_det)/total
    RCL = float(total-miss_det)/(total)
    if len(det)==0:
        RRL = 0
    else:
        RRL = float(total-miss_det)/len(det)
    return total, len(det), false_alarm, miss_det, FAR, MDR, RCL, RRL

def cluster_evlaution(ref, cluster):
    point_recog_0 = 0.0
    point_recog_0_ref_0 = 0.0
    point_recog_0_ref_1 = 0.0
    point_recog_1 = 0.0
    point_recog_1_ref_0 = 0.0
    point_recog_1_ref_1 = 0.0
    start = max(ref[0][0][0], cluster[0][0][0])
    end = min(ref[-1][0][1], cluster[-1][0][1])
    for i in range(time2frame(start, mfcc_shift), time2frame(end, mfcc_shift)):
        point = i/100.0
        if inpoint_t(point, cluster, 0):
            point_recog_0+=1
            if inpoint_t(point, ref, 0):
                point_recog_0_ref_0+=1
            elif inpoint_t(point, ref, 1):
                point_recog_0_ref_1+=1
            else:
                continue
        if inpoint_t(point, cluster, 1):
            point_recog_1+=1
            if inpoint_t(point, ref, 0):
                point_recog_1_ref_0+=1
            elif inpoint_t(point, ref, 1):
                point_recog_1_ref_1+=1
            else:
                continue
    p_0 = (point_recog_0_ref_0/point_recog_0)**2+(point_recog_0_ref_1/point_recog_0)**2
    p_1 = (point_recog_1_ref_0/point_recog_1)**2+(point_recog_1_ref_1/point_recog_1)**2
    acp = p_0*point_recog_0+p_1*point_recog_1
    acp_r = acp/(point_recog_0_ref_0+point_recog_0_ref_1+point_recog_1_ref_0+point_recog_1_ref_1)
    #print acp/(point_recog_0+point_recog_1)


    s_0 = (point_recog_0_ref_0/(point_recog_0_ref_0+point_recog_1_ref_0))**2+(point_recog_1_ref_0/(point_recog_0_ref_0+point_recog_1_ref_0))**2
    s_1 = (point_recog_0_ref_1/(point_recog_0_ref_1+point_recog_1_ref_1))**2+(point_recog_1_ref_1/(point_recog_0_ref_1+point_recog_1_ref_1))**2
    asp = s_0*(point_recog_0_ref_0+point_recog_1_ref_0)+s_1*(point_recog_0_ref_1+point_recog_1_ref_1)
    asp_r = asp/(point_recog_0_ref_0+point_recog_0_ref_1+point_recog_1_ref_0+point_recog_1_ref_1)
    return acp_r, asp_r

def gen_rttm(utt_name, ref, cluster):
    true_f = file('results_purity/init_segment/'+utt_name+'.ref.out', 'w')
    recog_f = file('results_purity/init_segment/'+utt_name+'.cog.out', 'w')
    start = max(ref[0][0][0], cluster[0][0][0])
    end = min(ref[-1][0][1], cluster[-1][0][1])
    for i in ref:
        if i[0][0]>start and i[0][1]<end:
            true_f.write(str(i[0][0])+' '+str(i[0][1])+' '+str(i[1])+'\n')

    for i in cluster:
        if i[0][0]>start and i[0][1]<end:
            recog_f.write(str(i[0][0])+' '+str(i[0][1])+' '+str(i[1])+'\n')
    # print ref
    # print cluster

#def cluster(utt_name, fout):
def gen_ref(filename):
    utt_c = []
    ref = []
    utt = file(filename).readlines()
    for i in utt:
        each = i[:-1].split()
        utt_c.append(each)

    for index, i in enumerate(utt_c):
        if index==0:
            continue
        if i[2]=='SIL' or i[2]=='OVERLAP':
            continue
        #print '----------',
        if utt_c[index-1][2]=='SIL':
            #print utt_c[index-1][0], utt_c[index-1][1]
            ref.append([float(utt_c[index-1][0]), float(utt_c[index-1][1])])
        else:
            #print utt_c[index-1][1], utt_c[index-1][1]
            ref.append([float(utt_c[index-1][1]), float(utt_c[index-1][1])])
        #print i
    return ref

def cluster(utt_name, fout):
    #featfile = 'data/'+utt_name+'/dvector.ark'
    vadfile = '../170309_thu_ev/data/'+utt_name+'/fbank_vad.ark'
    #ref = read_ref('ref/'+utt_name+'.txt')
    #ref_f = cpt_seg('ref/'+utt_name+'.txt')
    ref_f = gen_ref('../170309_thu_ev/thu_ev_tag/'+utt_name+'.txt')
    #print ref_f
    f_utt = 'data/'+utt_name
    
    vad_utt_label, vad_content = readvadfromkaldi(vadfile)
    vad = vad_content[0]
    
    feat = np.load("feat_pca/"+utt_name+".npy")
    
    feat_vad, feat_time = gen_feat_vad(feat, vad)
    
    #spk_model_gt = get_spk_model_ground_truth(ref, feat, spk_model_type)
    
    #print 'init_segmentation:'
    initial_det, initial_det_index = initial_seg(feat_vad, feat_time)
    init_e = seg_evlaution(initial_det, ref_f)
    print init_e
    #print initial_det
    #print initial_det_index
    #fout.write(utt_name+' ')
    #fout.write(str(init_e[0])+' '+str(init_e[1])+' '+str(init_e[2])+' '+str(init_e[3])+' '+str(init_e[4])+' '+str(init_e[5])+' ')
    
    # for i in zip(initial_det_index, initial_det):
    #     print i[0], i[1], feat_time[i[0]]
    #print 'init_cluster'
    cluster_change_point, cluster_result, spk_model = initial_cluster(initial_det_index, feat_vad, feat_time, spk_model_type)
    #print [frame2time(feat_time[i], mfcc_shift) for i in cluster_change_point]
    #print cluster_result
    cluster_e = seg_evlaution([frame2time(feat_time[i], mfcc_shift) for i in cluster_change_point], ref_f)
    print cluster_e

    #cluster_e_purity = cluster_evlaution(ref, [[[frame2time(feat_time[i[0][0]], mfcc_shift), frame2time(feat_time[i[0][1]], mfcc_shift)], i[1]] for i in cluster_result])
    #print cluster_e_purity

    #gen_rttm(utt_name, ref, [[[frame2time(feat_time[i[0][0]], mfcc_shift), frame2time(feat_time[i[0][1]], mfcc_shift)], i[1]] for i in cluster_result])
    
    #fout.write(utt_name+' '+str(cluster_e[0])+' '+str(cluster_e[1])+' '+str(cluster_e[2])+' '+str(cluster_e[3])+' '+str(cluster_e[4])+' '+str(cluster_e[5])+' '+str(cluster_e_purity[0])+' '+str(cluster_e_purity[1]))
    #fout.write(str(cluster_e[0])+' '+str(cluster_e[1])+' '+str(cluster_e[2])+' '+str(cluster_e[3])+' '+str(cluster_e[4])+' '+str(cluster_e[5]))
    #fout.write('\n')

def twospeaker_cluster(det_index, feat, feat_vad, feat_time, type, utt_name, num_segment):
    #spk0_rep, spk1_rep = [], []
    cluster_change_point, cluster_result = [], []
    last_index = 0
    block_rep, block_tag = gen_block_representation(det_index, feat_vad, feat_time)
    #print len(block_rep)
    #print block_tag


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
    '''
    for i in spk_dict[spk1][:num_segment]:
        spk1_data += feat[i[0]:i[1]].tolist()
    for i in spk_dict[spk2][:num_segment]:
        spk2_data += feat[i[0]:i[1]].tolist()
    '''
    for i in spk_dict[spk1]:
        spk1_data += feat[i[0]:i[1]].tolist()
    for i in spk_dict[spk2]:
        spk2_data += feat[i[0]:i[1]].tolist()
    spk0_rep = np.array(spk1_data)
    spk1_rep = np.array(spk2_data)
    
    '''    
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
    '''
    
    if type=='mean': 
        spk0_model = np.mean(spk0_rep, axis=0)
        spk1_model = np.mean(spk1_rep, axis=0)
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
        return cluster_change_point, cluster_result, [spk0_model, spk1_model]
    else:
        #clf = svm.SVC()
        #clf.fit(block_rep, y_pred)
        #return cluster_change_point, cluster_result, clf
        pass


def get_two_cluster(utt_name, fout):
    #featfile = 'data/'+utt_name+'/dvector.ark'
    vadfile = '../170309_thu_ev/data/'+utt_name+'/fbank_vad.ark'
    #ref = read_ref('ref/'+utt_name+'.txt')
    #ref_f = cpt_seg('ref/'+utt_name+'.txt')
    ref_f = gen_ref('../170309_thu_ev/thu_ev_tag/'+utt_name+'.txt')
    #print ref_f
    f_utt = 'data/'+utt_name
    
    vad_utt_label, vad_content = readvadfromkaldi(vadfile)
    vad = vad_content[0]
    
    feat = np.load("feat_pca/"+utt_name+".npy")
    
    feat_vad, feat_time = gen_feat_vad(feat, vad)
    
    #spk_model_gt = get_spk_model_ground_truth(ref, feat, spk_model_type)
    
    #print 'init_segmentation:'
    initial_det, initial_det_index = initial_seg(feat_vad, feat_time)
    init_e = seg_evlaution(initial_det, ref_f)
    #print init_e
    #print initial_det
    #print initial_det_index
    fout.write(utt_name+' ')
    fout.write(str(init_e[0])+' '+str(init_e[1])+' '+str(init_e[2])+' '+str(init_e[3])+' '+str(init_e[4])+' '+str(init_e[5])+' ')
    
    # for i in zip(initial_det_index, initial_det):
    #     print i[0], i[1], feat_time[i[0]]
    #print 'init_cluster'
    cluster_change_point, cluster_result, spk_model = twospeaker_cluster(initial_det_index, feat, feat_vad, feat_time, spk_model_type, utt_name, 5)
    #cluster_change_point, cluster_result, spk_model = initial_cluster(initial_det_index, feat_vad, feat_time, spk_model_type)
    #print [frame2time(feat_time[i], mfcc_shift) for i in cluster_change_point]
    #print cluster_result
    cluster_e = seg_evlaution([frame2time(feat_time[i], mfcc_shift) for i in cluster_change_point], ref_f)
    #print cluster_e

    #cluster_e_purity = cluster_evlaution(ref, [[[frame2time(feat_time[i[0][0]], mfcc_shift), frame2time(feat_time[i[0][1]], mfcc_shift)], i[1]] for i in cluster_result])
    #print cluster_e_purity

    #gen_rttm(utt_name, ref, [[[frame2time(feat_time[i[0][0]], mfcc_shift), frame2time(feat_time[i[0][1]], mfcc_shift)], i[1]] for i in cluster_result])
    
    #fout.write(utt_name+' '+str(cluster_e[0])+' '+str(cluster_e[1])+' '+str(cluster_e[2])+' '+str(cluster_e[3])+' '+str(cluster_e[4])+' '+str(cluster_e[5])+' '+str(cluster_e_purity[0])+' '+str(cluster_e_purity[1]))
    fout.write(str(cluster_e[0])+' '+str(cluster_e[1])+' '+str(cluster_e[2])+' '+str(cluster_e[3])+' '+str(cluster_e[4])+' '+str(cluster_e[5]))
    fout.write('\n')


#cluster('F001HJN_F002VAN_001','')
#get_two_cluster('F001HJN_F002VAN_001','')


fout = file('known_twospeaker_seg.out', 'w')
flist =  file('lst/thu_ev.lst').readlines()
for i in flist:
    each = i[:-1].split(' ')[0]
    print each
    get_two_cluster(each, fout)



