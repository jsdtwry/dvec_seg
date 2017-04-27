from initial_seg import *
from spk_cluster_reseg import *
import sys
import os

#wav_name = sys.argv[1]
#conf = sys.argv[2]
#nnet = sys.argv[3]

def sample_rate(old_name, new_name):
    os.system('sox '+old_name+' -r 8000 '+new_name)
    return

def feat_extract(wav_name, conf, nnet):
    utt_name = wav_name.split('/')[-1].split('.')[0]
    # mkdir
    if not os.path.exists('feat/'+utt_name):
        os.makedirs('feat/'+utt_name)
    # generate scp
    fout = file('feat/'+utt_name+'/wav.scp','w')
    fout.write(utt_name+' '+sys.path[0]+'/'+wav_name)
    fout.close()
    # feat_compute.sh
    # os.system('bash compute_feat.sh feat/'+utt_name+' '+conf+' '+nnet)
    data = 'feat/'+utt_name

    os.system('compute-fbank-feats --config='+conf+'/fbank.conf scp:'+data+'/wav.scp ark,t,scp:'+data+'/fbank_feats.ark,'+data+'/fbank_feats.scp')
    os.system('compute-vad --config='+conf+'/vad.conf scp:'+data+'/fbank_feats.scp ark,t,scp:'+data+'/fbank_vad.ark,'+data+'/fbank_vad.scp')
    os.system('nnet3-compute --use-gpu=no '+nnet+'/final.last_hid.raw ark:'+data+'/fbank_feats.ark ark,t,scp:'+data+'/dvector.ark,'+data+'/dvector.scp')

def dvec_seg(wav_name):
    utt_name = wav_name.split('/')[-1].split('.')[0]
    dvector_file = 'feat/'+utt_name+'/dvector.ark' # 400 dim
    vad_file = 'feat/'+utt_name+'/fbank_vad.ark'
    
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
    cluster_result = [[[frame2time(feat_time[i[0][0]], mfcc_shift), frame2time(feat_time[i[0][1]], mfcc_shift)], int(i[1])] for i in segment_result]
    
    return cluster_result
  
def dvec_seg_one_model(wav_name, start, end):
    utt_name = wav_name.split('/')[-1].split('.')[0]
    dvector_file = 'feat/'+utt_name+'/dvector.ark' # 400 dim
    vad_file = 'feat/'+utt_name+'/fbank_vad.ark'
    
    # load d-vector feature and vad result
    utt_lable, feat_content = readfeatfromkaldi(dvector_file, 400)
    vad_utt_label, vad_content = readvadfromkaldi(vad_file)
    
    # remove silence regions
    feat_vad, feat_time = gen_feat_vad(feat_content, vad_content)
    
    # initial segmentation
    scores, times, det_time, det_index = initial_seg(feat_vad, feat_time, 0.1, 0.01, 'dvec')

    # k-means clustering from initial segmentation
    #change_point, segment_result, spk_model = spk_k_means_cluster(det_index, feat_vad, feat_time, 'svm')
    spk_one = np.mean(feat_content[int(start*100):int(end*100)], axis=0)
    #print len(spk_one)
    #spk_model = get_spk_model(feat_content, 'F001HJN_F002VAN_001')
    change_point, segment_result = spk_reseg_with_one_model(det_index, feat_vad, feat_time, spk_one)
    # transform index of features to time points
    det_time = [frame2time(feat_time[i], mfcc_shift) for i in change_point]
    cluster_result = [[[frame2time(feat_time[i[0][0]], mfcc_shift), frame2time(feat_time[i[0][1]], mfcc_shift)], i[1]] for i in segment_result]
    return cluster_result


#feat_extract(wav_name, conf, nnet)
#l = dvec_seg(wav_name)
#print l
#dvec_seg_one_model(wav_name, 0.21, 1.50)

