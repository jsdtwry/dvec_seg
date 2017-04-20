from spk_cluster_reseg import *
from evaluation import *
from initial_seg import *
import sys
import os

def pre_processing():
    uttlst = file('lst/t2.lst').readlines()
    for i in uttlst:
        each = i[:-1].split()
        os.makedirs('data/'+each[0])
        fout = file('data/'+each[0]+'/wav.scp','w')
        fout.write(i)

def feat_extraction():
    uttlst = file('lst/t2.lst').readlines()
    for i in uttlst:
        each = i[:-1].split()
        os.system('bash fea_extract.sh data/'+each[0])
    print 'pre-processing and feature extraction done!'
    

def dvec_demo():
    dvector_file = 'data/F001HJN_F002VAN_001/dvector.ark' # 400 dim
    vad_file = 'data/F001HJN_F002VAN_001/fbank_vad.ark'
    
    # load d-vector feature and vad result
    utt_lable, feat_content = readfeatfromkaldi(dvector_file, 400)
    vad_utt_label, vad_content = readvadfromkaldi(vad_file)
    
    # remove silence regions
    feat_vad, feat_time = gen_feat_vad(feat_content, vad_content)
    
    # initial segmentation
    scores, times, det_time, det_index = initial_seg(feat_vad, feat_time, 0.1, 0.01, 'dvec')

    # k-means clustering from initial segmentation
    change_point, segment_result, spk_model = spk_k_means_cluster(det_index, feat_vad, feat_time, 'svm')

    # transform index of features to time points
    det_time = [frame2time(feat_time[i], mfcc_shift) for i in change_point]
    cluster_result = [[[frame2time(feat_time[i[0][0]], mfcc_shift), frame2time(feat_time[i[0][1]], mfcc_shift)], i[1]] for i in segment_result]

    # load change points labels and speaker labels
    ref_segment = gen_ref_seg('thu_ev_tag/F001HJN_F002VAN_001.txt')
    ref = read_ref('thu_ev_tag/F001HJN_F002VAN_001.txt')
    
    # evlaution
    init_e = seg_evlaution(det_time, ref_segment, 0.3)
    cluster_e = cluster_evluation(ref, cluster_result)
    print 'K-means clustering:'
    print init_e
    print cluster_e
    
    # resegmentation with speaker models
    spk_model = get_spk_model(feat_content, 'F001HJN_F002VAN_001')
    change_point, segment_result = spk_reseg_with_models(det_index, feat_vad, feat_time, spk_model)
    det_time = [frame2time(feat_time[i], mfcc_shift) for i in change_point]
    cluster_result = [[[frame2time(feat_time[i[0][0]], mfcc_shift), frame2time(feat_time[i[0][1]], mfcc_shift)], i[1]] for i in segment_result]

    # evlaution
    init_e = seg_evlaution(det_time, ref_segment, 0.3)
    cluster_e = cluster_evluation(ref, cluster_result)
    print 'resegmentation with speaker models:'
    print init_e
    print cluster_e
 

def bic_change_det():
    mfcc_file = 'data/F001HJN_F002VAN_001/mfcc_feats.ark' # 20 dim
    vad_file = 'data/F001HJN_F002VAN_001/mfcc_vad.ark'
    
    # load MFCC feature and vad result
    utt_lable, feat_content = readfeatfromkaldi(mfcc_file, 20)
    vad_utt_label, vad_content = readvadfromkaldi(vad_file)

    # remove silence regions
    feat_vad, feat_time = gen_feat_vad(feat_content, vad_content)
    
    # initial segmentation
    scores, times, det_time, det_index = initial_seg(feat_vad, feat_time, 1, 0.1, 'bic', lamda=1.0)
    print times

    # fixed threshold segmentation
    det_tmp = fix_slid_det_bic(times, scores, 0)
    print det_tmp

def det_eer(lst_filename, seg_type, ft):
    full_scores, scores_c, times_c, ref_c = [], [], [], []
    eer_x, eer_y, eer_t = [], [], []
    for i in file(lst_filename).readlines():
        utt = i.split()[0]
        if seg_type=='dvec':
            scores, times, det_time, det_index = initial_segmentation('data/'+utt+'/dvector.ark', 'data/'+utt+'/fbank_vad.ark', 400, 0.1, 0.01, seg_type)
        else:
            scores, times, det_time, det_index = initial_segmentation('data/'+utt+'/mfcc_feats.ark', 'data/'+utt+'/mfcc_vad.ark', 20, 1, 0.1, seg_type)
        ref_segment = gen_ref_seg('thu_ev_tag/'+utt+'.txt')
        scores_c.append(scores)
        times_c.append(times)
        ref_c.append(ref_segment)
        full_scores.extend(scores)
    thr = np.linspace(min(full_scores), max(full_scores), 100)
    for T in thr:
        false_alarm, miss_det, total = 0, 0, 0
        for index, scores in enumerate(scores_c):
            if seg_type=='dvec':
                det_tmp = fix_slid_det_dvec(times_c[index], scores, float(T))
            else:
                det_tmp = fix_slid_det_bic(times_c[index], scores, float(T))
            det = []
            for i in det_tmp:
                if i > ref_c[index][0][0] and i < ref_c[index][-1][-1]:
                    det.append(i)
            for i in det:
                if not inline(float(i), ref_c[index], ft):
                    false_alarm += 1
            for i in ref_c[index]:
                if not inpoint(i, det, ft):
                    miss_det += 1

            total += len(ref_c[index])
        FAR = float(false_alarm)/(total+false_alarm)
        MDR = float(miss_det)/total
        print T,
        print FAR,
        print MDR
	#print false_alarm, miss_det, total
        eer_x.append(FAR)
        eer_y.append(MDR)
        eer_t.append(T)
    dist = []
    for index, each in enumerate(eer_x):
        dist.append(abs(each - eer_y[index]))
    ii = dist.index(min(dist))
    print eer_x[ii], eer_y[ii], eer_t[ii]


#pre_processing()
#feat_extraction()

det_eer('lst/t2.lst', 'bic', 0.3)
bic_change_det()
dvec_demo()
