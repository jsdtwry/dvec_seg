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

'''
generate reference of change segments
'''
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

'''
generate reference speaker labels
'''
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
Given clustering result and reference speaker labels to calculate the ACP, ASP
'''
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
    if point_recog_0==0:
        p_0 = 0 
    else:
        p_0 = (point_recog_0_ref_0/point_recog_0)**2+(point_recog_0_ref_1/point_recog_0)**2
    if point_recog_1==0:
        p_1 = 0 
    else:
        p_1 = (point_recog_1_ref_0/point_recog_1)**2+(point_recog_1_ref_1/point_recog_1)**2
    acp = p_0*point_recog_0+p_1*point_recog_1
    acp_r = acp/(point_recog_0_ref_0+point_recog_0_ref_1+point_recog_1_ref_0+point_recog_1_ref_1)
    
    if point_recog_0_ref_0+point_recog_1_ref_0==0:
        s_0 = 0 
    else:
        s_0 = (point_recog_0_ref_0/(point_recog_0_ref_0+point_recog_1_ref_0))**2+(point_recog_1_ref_0/(point_recog_0_ref_0+point_recog_1_ref_0))**2
    if point_recog_0_ref_1+point_recog_1_ref_1==0:
        s_1 = 0 
    else:
        s_1 = (point_recog_0_ref_1/(point_recog_0_ref_1+point_recog_1_ref_1))**2+(point_recog_1_ref_1/(point_recog_0_ref_1+point_recog_1_ref_1))**2
    asp = s_0*(point_recog_0_ref_0+point_recog_1_ref_0)+s_1*(point_recog_0_ref_1+point_recog_1_ref_1)
    asp_r = asp/(point_recog_0_ref_0+point_recog_0_ref_1+point_recog_1_ref_0+point_recog_1_ref_1)
    return acp_r, asp_r

