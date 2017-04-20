from spk_cluster_reseg import *
from evaluation import *
from initial_seg import *

#scores, times, det_time, det_index = initial_segmentation('data/F001HJN_F002VAN_001/mfcc_feats.ark', 'data/F001HJN_F002VAN_001/fbank_vad.ark', 20, 1, 0.1, 'bic')

mfcc_file = 'data/F001HJN_F002VAN_001/mfcc_feats.ark' # 20 dim
dvector_file = 'data/F001HJN_F002VAN_001/dvector.ark' # 400 dim
vad_file = 'data/F001HJN_F002VAN_001/fbank_vad.ark'

utt_lable, feat_content = readfeatfromkaldi(dvector_file, 40)
#print utt_lable
#print len(content)
#print content[0]

vad_utt_label, vad_content = readvadfromkaldi(vad_file)

#print vad_utt_label
#print len(vad_content)
#print vad_content[0]

ref_segment = gen_ref_seg('thu_ev_tag/F001HJN_F002VAN_001.txt')
ref = read_ref('thu_ev_tag/F001HJN_F002VAN_001.txt')

# initial segmentation
feat_vad, feat_time = gen_feat_vad(feat_content, vad_content)
scores, times, det_time, det_index = initial_seg(feat_vad, feat_time, 0.1, 0.01, 'dvec')
#scores, times, det_time, det_index = initial_seg(feat_vad, feat_time, 1, 0.1, 'bic', lamda=1.0)

init_e = seg_evlaution(det_time, ref_segment, 0.3)
print init_e



# resegmentation with k-means
change_point, segment_result, spk_model = spk_k_means_cluster(det_index, feat_vad, feat_time, 'svm')

det_time_1 = [frame2time(feat_time[i], mfcc_shift) for i in change_point]
cluster_result = [[[frame2time(feat_time[i[0][0]], mfcc_shift), frame2time(feat_time[i[0][1]], mfcc_shift)], i[1]] for i in segment_result]

init_e = seg_evlaution(det_time_1, ref_segment, 0.3)
print init_e

cluster_e = cluster_evluation(ref, cluster_result)
print cluster_e


# resegmentation with speaker models
spk_model = get_spk_model(feat_content, 'F001HJN_F002VAN_001')
change_point, segment_result = spk_reseg_with_models(det_index, feat_vad, feat_time, spk_model)
det_time_1 = [frame2time(feat_time[i], mfcc_shift) for i in change_point]
cluster_result = [[[frame2time(feat_time[i[0][0]], mfcc_shift), frame2time(feat_time[i[0][1]], mfcc_shift)], i[1]] for i in segment_result]

init_e = seg_evlaution(det_time_1, ref_segment, 0.3)
print init_e

cluster_e = cluster_evluation(ref, cluster_result)
print cluster_e

'''
detlist_eer('lst/thu_ev.lst', 'dvec', 0.3)
detlist_eer('lst/thu_ev.lst', 'dvec', 0.2)
detlist_eer('lst/thu_ev.lst', 'dvec', 0.1)
'''


