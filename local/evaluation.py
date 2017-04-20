import numpy as np
import initial_seg

mfcc_len = 0.10
mfcc_shift = 0.01

def time2frame(time, mfcc_shift):
    return int(time*1/mfcc_shift)

def frame2time(frame, mfcc_shift):
    return float(frame/(1/mfcc_shift))

def inpoint_t(point, list_tag, tag):
    for i in list_tag:
        if i[1]==tag and point>=i[0][0] and point <= i[0][1]:
            return True
    return False


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

def gen_ref_seg(filename):
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
        if utt_c[index-1][2]=='SIL':
            ref.append([float(utt_c[index-1][0]), float(utt_c[index-1][1])])
        else:
            ref.append([float(utt_c[index-1][1]), float(utt_c[index-1][1])])
    return ref

def read_ref(filename):
    utt_c = []
    ref = []
    utt = file(filename).readlines()
    for i in utt:
        each = i[:-1].split()
        utt_c.append(each)
    flag = 0
    for index, i in enumerate(utt_c):
        if i[2]=='SIL' or i[2]=='OVERLAP':
            continue
        if flag == 0:
            spk_1 = i[2]
            flag = 1
            ref.append([[float(i[0]),float(i[1])], 1])
        else:
            if spk_1==i[2]:
                ref.append([[float(i[0]),float(i[1])], 1])
            else:
                ref.append([[float(i[0]),float(i[1])], 0])
    return ref
'''
Given detected change points, reference change segments and the torlance to calculate FAR, MDR, RCL, RRL
'''
def seg_evlaution(det_tmp, ref_f, ft):
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
    total = len(ref_f)
    FAR = float(false_alarm)/(total+false_alarm)
    MDR = float(miss_det)/total
    RCL = float(total-miss_det)/(total)
    if len(det)==0:
        RRL = 0
    else:
        RRL = float(total-miss_det)/len(det)
    return total, len(det), false_alarm, miss_det, FAR, MDR, RCL, RRL

'''
generate det curves with all files in the list
'''
def detlist_eer(lst_filename, seg_type, ft):
    full_scores, scores_c, times_c, ref_c = [], [], [], []
    eer_x, eer_y, eer_t = [], [], []
    for i in file(lst_filename).readlines():
        utt = i.split()[0]
        scores, times, det_time, det_index = initial_seg.initial_segmentation('../170309_thu_ev/data/'+utt+'/dvector.ark', '../170309_thu_ev/data/'+utt+'/fbank_vad.ark', 400, 0.1, 0.01, seg_type)
        ref_segment = gen_ref_seg('../170309_thu_ev/thu_ev_tag/'+utt+'.txt')
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
    print seg_type, '======================='
    print eer_x[ii], eer_y[ii], eer_t[ii]

def cluster_evluation(ref, cluster):
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

    s_0 = (point_recog_0_ref_0/(point_recog_0_ref_0+point_recog_1_ref_0))**2+(point_recog_1_ref_0/(point_recog_0_ref_0+point_recog_1_ref_0))**2
    s_1 = (point_recog_0_ref_1/(point_recog_0_ref_1+point_recog_1_ref_1))**2+(point_recog_1_ref_1/(point_recog_0_ref_1+point_recog_1_ref_1))**2
    asp = s_0*(point_recog_0_ref_0+point_recog_1_ref_0)+s_1*(point_recog_0_ref_1+point_recog_1_ref_1)
    asp_r = asp/(point_recog_0_ref_0+point_recog_0_ref_1+point_recog_1_ref_0+point_recog_1_ref_1)
    return acp_r, asp_r


'''
ref_segment = gen_ref_seg('../170309_thu_ev/thu_ev_tag/F001HJN_F002VAN_001.txt')
scores, times, det_time, det_index = initial_seg.initial_segmentation('data/F001HJN_F002VAN_001/mfcc_feats.ark', 'data/F001HJN_F002VAN_001/fbank_vad.ark', 20, 1, 0.1, 'bic')

#init_e = seg_evlaution(det_time, ref_segment, 0.3)
print init_e
'''

# detlist_eer('lst/thu_ev.lst', 'bic', 0.3)


