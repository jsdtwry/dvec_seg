import numpy as np
import math
import scipy
from sklearn import mixture

mfcc_len = 0.10
mfcc_shift = 0.01

def time2frame(time, mfcc_shift):
    return int(time*1/mfcc_shift)

def frame2time(frame, mfcc_shift):
    return float(frame/(1/mfcc_shift))

'''
Given the kaldi feats file and the feat dimension and read to the numpy array
'''
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

	return utt_lable, np.array(content[0])

'''
Given the kaldi vad file and the feat dimension and read to the numpy array
'''
def readvadfromkaldi(filename):
	vb = file(filename).readlines()
	utt_lable = []
	content = []
	utt_content = []
	for i in vb:
		vad = i.split(' ')
		utt_lable.append(vad[0])
		content.append(vad[3:-1])
	return utt_lable, np.array(content[0])

'''
Remove slience blocks from feats
'''
def gen_feat_vad(feat, vad):
    new_feat = []
    feat_time = []
    for index, i in enumerate(zip(feat, vad)):
        if i[1]!='0':
            new_feat.append(i[0])
            feat_time.append(index)
    return new_feat, feat_time

'''
calculate cosine distance of two d-vectors
'''
def dvec_cos(left, right):
    leftvec = np.mean(np.array(left), axis=0)
    rightvec = np.mean(np.array(right), axis=0)
    cos = np.dot(leftvec,rightvec)/(np.linalg.norm(leftvec)*np.linalg.norm(rightvec))
    return cos

'''
calculate bic distance
'''
def bic_dist(left, right, full, lamda):
    cov_l = np.cov(np.array(left).T)
    cov_r = np.cov(np.array(right).T)
    cov_f = np.cov(np.array(full).T)
    len_l = len(left)
    len_r = len(right)
    len_f = len(full)
    d = len(cov_f)
    r = len_f*math.log(np.linalg.det(cov_f)) - len_l*math.log(np.linalg.det(cov_l)) - len_r*math.log(np.linalg.det(cov_r))
    score = 0.5*(r-(d+0.5*d*(d+1))*math.log(float(len_f))*lamda)
    return score

'''
calculate kl distance
'''
def kl(left, right):
    gaussian_l = mixture.GMM(n_components=1, covariance_type='full')
    gaussian_l.fit(left)
    gaussian_r = mixture.GMM(n_components=1, covariance_type='full')
    gaussian_r.fit(right)
    sigma_l = gaussian_l.covars_[0]
    mean_l = gaussian_l.means_[0]
    sigma_r = gaussian_r.covars_[0]
    mean_r = gaussian_r.means_[0]
    kl1 = 0.5 * (np.trace((sigma_l - sigma_r)*(np.linalg.inv(sigma_l) -np.linalg.inv(sigma_r))) + np.trace((np.linalg.inv(sigma_l) - np.linalg.inv(sigma_r))*((mean_l - mean_r)*(mean_l - mean_r).T)))
    kl2 = 0.5 * (np.trace((sigma_r - sigma_l)*(np.linalg.inv(sigma_r) -np.linalg.inv(sigma_l))) + np.trace((np.linalg.inv(sigma_r) - np.linalg.inv(sigma_l))*((mean_r - mean_l)*(mean_r - mean_l).T)))
    return -(kl1+kl2)

'''
calculate glr distance
'''
def glr(left, right, full):
    gaussian_l = mixture.GMM(n_components=1, covariance_type='full')
    gaussian_l.fit(left)
    gaussian_r = mixture.GMM(n_components=1, covariance_type='full')
    gaussian_r.fit(right)
    gaussian_f = mixture.GMM(n_components=1, covariance_type='full')
    gaussian_f.fit(full)
    gaussian_l_score = scipy.log(-gaussian_l.score(left))
    gaussian_r_score = scipy.log(-gaussian_r.score(right))
    gaussian_f_score = scipy.log(-gaussian_f.score(full))
    glr = sum(gaussian_f_score) - (sum(gaussian_l_score) + sum(gaussian_r_score))
    return glr

'''
'''
def fix_slid_det_dvec_nothreshold(times, times_index, scores):
    det, det_index = [], []
    for index, each in enumerate(scores):
            if index == 0 or index == len(scores)-1:
                    continue
            if each < scores[index-1] and each < scores[index+1]:
                    det.append(times[index])
                    det_index.append(times_index[index])
    return det, det_index

def fix_slid_det_bic_nothreshold(times, times_index, scores):
    det, det_index = [], []
    for index, each in enumerate(scores):
            if index == 0 or index == len(scores)-1:
                    continue
            if each > scores[index-1] and each > scores[index+1]:
                    det.append(times[index])
                    det_index.append(times_index[index])
    return det, det_index


'''Init segmentation with no threshold return the list of change point and change point reaf'''
'''[1.92, 2.16, 2.65, 2.78, 2.88, 3.14]'''
def initial_seg(feat_vad, feat_time, win_len, win_shift, dist, lamda=1.5):
    scores, times, times_index = [], [], []
    f_len = len(feat_vad)
    startpoint = 0
    while startpoint + time2frame(win_len*2, mfcc_shift) < f_len:
        mid = startpoint + time2frame(win_len, mfcc_shift)
        left_features = feat_vad[mid-time2frame(win_len, mfcc_shift):mid]
        right_features = feat_vad[mid:mid+time2frame(win_len, mfcc_shift)]
        full_features = feat_vad[mid-time2frame(win_len, mfcc_shift):mid+time2frame(win_len, mfcc_shift)]
        if dist=='dvec': 
            scores.append(dvec_cos(left_features, right_features))
        if dist=='bic':
		    scores.append(bic_dist(left_features, right_features, full_features, lamda))
        if dist=='kl':
            scores.append(kl(left_features, right_features))
        if dist=='glr':
            scores.append(glr(left_features, right_features, full_features))
        times.append(frame2time(feat_time[mid], mfcc_shift))
        times_index.append(mid)
        startpoint += time2frame(win_shift, mfcc_shift)
    if dist=='dvec':
        det_time, det_index = fix_slid_det_dvec_nothreshold(times, times_index, scores)
    else:
        det_time, det_index = fix_slid_det_bic_nothreshold(times, times_index, scores)
    return scores, times, det_time, det_index


def initial_segmentation(feat_filename, vad_filename, feat_dim, win_size, win_shift, seg_type, lamda=1.5):
    utt_lable, content = readfeatfromkaldi(feat_filename, feat_dim)
    vad_utt_label, vad_content = readvadfromkaldi(vad_filename)
    feat_vad, feat_time = gen_feat_vad(content, vad_content)
    return initial_seg(feat_vad, feat_time, win_size, win_shift, seg_type)
'''
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
scores, times, det_time, det_index = initial_seg(feat_vad, feat_time, 0.1, 0.01, 'dvec')
scores, times, det_time, det_index = initial_seg(feat_vad, feat_time, 1, 0.1, 'bic', lamda=1.0)

print len(scores), len(times), len(det_time), len(det_index)
print det_time
print det_index
print times
#initial_seg(feat_vad, feat_time, 1, 0.1, 'glr')
#initial_seg(feat_vad, feat_time, 1, 0.1, 'kl')

scores, times, det_time, det_index = initial_segmentation('data/F001HJN_F002VAN_001/mfcc_feats.ark', 'data/F001HJN_F002VAN_001/fbank_vad.ark', 20, 1, 0.1, 'bic')
print len(scores), len(times), len(det_time), len(det_index)
'''




