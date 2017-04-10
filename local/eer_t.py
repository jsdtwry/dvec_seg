import sys
import numpy as np


lstfile = sys.argv[1]

ft = 0.1

def fix_slid_det_bic(times, scores, threshold):
        det = []
        for index, each in enumerate(scores):
                if index == 0 or index == len(scores)-1:
                        continue
                if each > scores[index-1] and each > scores[index+1] and each > threshold:
                        det.append(times[index])
        return det

def fix_slid_det_dvec(times, scores, threshold):
        det = []
        for index, each in enumerate(scores):
                if index == 0 or index == len(scores)-1:
                        continue
                if each < scores[index-1] and each < scores[index+1] and each < threshold:
                        det.append(times[index])
        return det

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

def gen_ref(filename):
    utt_c = []
    ref = []
    utt = file(filename).readlines()
    for i in utt:
        each = i[:-1].split(' ')
        utt_c.append(each)

    for index, i in enumerate(utt_c):
        if index==0:
            continue
        if i[2]=='SIL':
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

def eer(flst, seg_type):
	times_c = []
	scores_c = []
	ref_c = []
	scores_full = []
	eer_x = []
	eer_y = []
	eer_t = []
	for f_utt in flst:
		ltimes = file(f_utt+'/'+seg_type+'_time.tag').readlines()
		times_t = ltimes[0][:-1].split(' ')
		times_f = [float(i) for i in times_t]
		lscores = file(f_utt+'/'+seg_type+'_score.tag').readlines()
		scores_t = lscores[0][:-1].split(' ')
		scores_f = [float(i) for i in scores_t]
		#lref = file(f_utt+'/'+seg_type+'_ref.tag').readlines()
		#ref_f = []
		#for ref_seg in lref:
		#	lref_seg = ref_seg[:-1].split(' ')
		#	lref_seg_f = [float(lref_seg[0]), float(lref_seg[1])]
		#	ref_f.append(lref_seg_f)
        #print 'thu_ev_tag/'+f_utt.split('/')[1]+'.txt'
        ref_f = gen_ref('thu_ev_tag/'+f_utt.split('/')[1]+'.txt')
        times_c.append(times_f)
        scores_c.append(scores_f)
        ref_c.append(ref_f)
        scores_full.extend(scores_f)

	#print len(times_c)
	#print len(scores_c)
	#print len(ref_c)
	#print len(scores_full)
	#print times_c
	#print scores_c
	#print ref_c

	thr = np.linspace(min(scores_full), max(scores_full), 100)
	for T in thr:
        	false_alarm = 0
        	miss_det = 0
        	total = 0
        	for index, scores in enumerate(scores_c):
			if seg_type=='bic':
                		det_tmp = fix_slid_det_bic(times_c[index], scores, float(T))
			elif seg_type=='glr':
				det_tmp = fix_slid_det_bic(times_c[index], scores, float(T))
			elif seg_type=='kl':
				det_tmp = fix_slid_det_bic(times_c[index], scores, float(T))
			else:
				det_tmp = fix_slid_det_dvec(times_c[index], scores, float(T))
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
		#print false_alarm, miss_det, total
        	eer_x.append(FAR)
        	eer_y.append(MDR)
        	eer_t.append(T)
	dist = []
	for index, each in enumerate(eer_x):
        	dist.append(abs(each - eer_y[index]))
	ii = dist.index(min(dist))
	print seg_type
	print eer_x[ii], eer_y[ii], eer_t[ii]



lst = file(lstfile).readlines()
utt_list = ['data/'+i[:-1].split(' ')[0] for i in lst]
#print utt_list
#eer(['data/fe_03_05954/'],'bic')
#eer(['data/fe_03_05954/'],'dvec')
#eer(utt_list,'bic')
eer(utt_list,'bic')
#eer(utt_list,'dvec')
