import sys
import numpy as np
import math
from scipy import signal
#import matplotlib
#import matplotlib.pyplot as plt
'''
featfile = sys.argv[1]
vadfile = sys.argv[2]
win_len = sys.argv[3]
win_shift = sys.argv[4]
lamda = sys.argv[5]
threshold = sys.argv[6]
'''
u_utt = sys.argv[1]
win_len = float(sys.argv[2])
win_shift = float(sys.argv[3])
featfile = u_utt+'/mfcc_feats.ark'
vadfile = u_utt+'/mfcc_vad.ark'
lamda = 1.5
threshold = -220


mfcc_len = 0.02
mfcc_shift = 0.01

ft = 0.3

def inline(point, ref, ft):
	for i in ref:
		if point >= i[0]-ft and point <= i[1]+ft:
			return True
	return False
def inpoint(seg, det, ft):
	#print seg
	#print det
	for i in det:
		if float(i) >= seg[0]-ft and float(i) <= seg[1]+ft:
			return True
	return False


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
				#print '----'
				#print [float(utt_t[index-2].split(' ')[1]), start]
				cpt_seg.append(sorted([float(utt_t[index-2].split(' ')[1]), start]))
				#cpt_seg.append([float(utt_t[index-2].split(' ')[1]), start])
			#print index, start, end, spk
	return cpt_seg

def cal_rate(ref, det, ft):
	#print ref
	#print det[1]
	false_alarm = 0
	miss_det = 0
	for i in det:
		if not inline(float(i), ref, ft):
			false_alarm += 1

	for i in ref:
		if not inpoint(i, det, ft):
			miss_det += 1

	return  len(ref), len(det), false_alarm, miss_det

def time2frame(time, mfcc_shift):
	return int(time*1/mfcc_shift)

def frame2time(frame, mfcc_shift):
	return float(frame/(1/mfcc_shift))

def bic(left, right, full , lamda):
    #print left
    #print right
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

def get_utt_vad(utt, vad):
	utt_vad = []
	utt_vad_index = []
	for index, vadlable in enumerate(vad):
		if vadlable=='1':
			utt_vad_index.append(index)
			utt_vad.append(utt[index])
	return utt_vad, utt_vad_index

def fix_slid(utt_vad, utt_vad_index, win_len, win_shift, lamda):
	times = []
	scores = []
	f_len = len(utt_vad)
	startpoint = 0
	win_len_frame = time2frame(win_len, mfcc_shift)
	while startpoint + win_len_frame*2 < f_len:
		mid = startpoint + win_len_frame
		left_features = utt_vad[mid-win_len_frame:mid]
		right_features = utt_vad[mid:mid+win_len_frame]
		full_features = utt_vad[mid-win_len_frame:mid+win_len_frame]
		scores.append(bic(left_features, right_features, full_features, lamda))
		times.append(frame2time(utt_vad_index[mid], mfcc_shift))
		startpoint += time2frame(win_shift, mfcc_shift)
		#print mid-win_len_frame, mid, mid+win_len_frame
	return times, scores

def fix_slid_det(times, scores, threshold):
	det = []
	for i,j in enumerate(scores):
		if j > threshold:
			det.append(times[i])
	return det

def fix_slid_det_2(times, scores, threshold):
	det = []
	peakind = signal.find_peaks_cwt(np.array(scores), np.arange(1,10))
	for i in peakind:
		if scores[i] > threshold:
			det.append(times[i])
	return det

def fix_slid_det_3(times, scores, threshold):
	det = []
	for index, each in enumerate(scores):
		if index == 0 or index == len(scores)-1:
			continue
		if each > scores[index-1] and each > scores[index+1] and each > threshold:
			det.append(times[index])
	return det

def fix_slid_det_4(times, scores, threshold):
	det = []
	for index, each in enumerate(scores):
		if index == 0 or index == len(scores)-1:
			continue
		if each > scores[index-1] and each > scores[index+1] and each > threshold:
			det.append(times[index])
	return det
'''
def roc(scores, ref, ft):
	roc_x = []
	roc_y = []
	thr = np.linspace(min(scores), max(scores), 100)
	for T in thr:
		false_alarm = 0
		miss_det = 0
		det = fix_slid_det_3(times, scores, float(T))
		for i in det:
			if not inline(float(i), ref, ft):
				false_alarm += 1

		for i in ref:
			if not inpoint(i, det, ft):
				miss_det += 1

		total = len(ref)
		FAR = float(false_alarm)/(total+false_alarm)
		MDR = float(miss_det)/total
		#print T
		#print FAR,
		#print MDR
		roc_x.append(FAR)
		roc_y.append(MDR)
	dist = []
	for index, each in enumerate(roc_x):
		dist.append(abs(each - roc_y[index]))
	ii = dist.index(min(dist))
	print roc_x[ii], roc_y[ii]

	plt.plot(roc_x, roc_y)
	#plt.plot([0, 1], [0, 1], '--', color='red')
	plt.xlabel("False Alarm(%)")
	plt.ylabel("Miss Detection(%)")
	plt.xticks(np.linspace(0, 1, 11))
	plt.yticks(np.linspace(0, 1, 11))
	plt.grid(True)
	#plt.show()

'''
utt_lable, content = readfeatfromkaldi(featfile, 20)
#print utt_lable
print len(content)
print len(content[0])

vad_utt_label, vad_content = readvadfromkaldi(vadfile)

#total = 0
#det = 0
#false_alarm = 0
#miss_det = 0

times_c = []
scores_c = []
#ref_c = []
eer_x = []
eer_y = []
eer_t = []
for index, utt in enumerate(content):
	utt_vad, utt_vad_index = get_utt_vad(utt, vad_content[index])
	times, scores = fix_slid(utt_vad, utt_vad_index, float(win_len), float(win_shift), float(lamda))
	#ref = cpt_seg('/nfs/user/wangrenyu/data/fisher_seg_sel_059/059_ref/'+utt_lable[index]+'.txt')
	times_c.append(times)
	scores_c.append(scores)
	#ref_c.append(ref)

ftimes = file(u_utt+'/bic_time.tag','w')
for i in times_c[0]:
	ftimes.write(str(i)+' ')

fscores = file(u_utt+'/bic_score.tag','w')
for i in scores_c[0]:
	fscores.write(str(i)+' ')

#fref = file(u_utt+'/bic_ref.tag','w')
#for i in ref_c[0]:
#	fref.write(str(i[0])+' '+str(i[1])+'\n')
#print len(times_c[0])
'''
thr = np.linspace(min(scores), max(scores), 100)
for T in thr:
	false_alarm = 0
	miss_det = 0
	total = 0
	for index, scores in enumerate(scores_c):
		det_tmp = fix_slid_det_3(times_c[index], scores, float(T))
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
	print T
	print FAR,
	print MDR
	eer_x.append(FAR)
	eer_y.append(MDR)
	eer_t.append(T)
dist = []
for index, each in enumerate(eer_x):
	dist.append(abs(each - eer_y[index]))
ii = dist.index(min(dist))
print 'bic:',win_len, win_shift
print eer_x[ii], eer_y[ii], eer_t[ii]
'''
#print vad_utt_label
#print len(vad_content[0])

#utt_vad, utt_vad_index = utt_vad(utt_lable, content[0], vad_content[0])
#print len(utt_vad)
#print len(utt_vad_index) 

#times, scores = fix_slid(utt_vad, utt_vad_index, float(win_len), float(win_shift), float(lamda))
#print scores

#dets = fix_slid_det_3(times, scores, float(threshold))
#ref = cpt_seg('ref/fe_03_00002.txt')
#print ref
'''
for i in ref:
	plt.axvspan(i[0], i[1], facecolor='g', alpha=0.5)

plt.plot(times, scores, 'o-', linewidth=1, marker='o', markersize=2)
plt.show()
'''
'''
print ref
print dets

total, det, false_alarm, miss_det = cal_rate(ref, dets, ft)
FAR = float(false_alarm)/(total+false_alarm)
MDR = float(miss_det)/total
print FAR, MDR
'''

#roc(scores, ref, ft)
'''

for i in [0.01,0.05,0.1,0.2,0.3,0.5]:
	times, scores = fix_slid(utt_vad, utt_vad_index, 2.0, i, float(lamda))
	roc(scores, ref, ft)

plt.legend(('0.01 s', '0.05 s', '0.1 s', '0.2 s', '0.3 s', '0.5s '), shadow=True, loc=(0.7, 0.7))
ltext = plt.gca().get_legend().get_texts()
plt.plot([0, 1], [0, 1], '--', color='red')
plt.show()

'''
