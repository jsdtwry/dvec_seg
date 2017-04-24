data=$1
conf=$2
nnet=$3
#compute-mfcc-feats --config=conf/mfcc.conf scp:$data/wav.scp ark,t,scp:$data/mfcc_feats.ark,$data/mfcc_feats.scp
#compute-vad --config=conf/vad.conf scp:$data/mfcc_feats.scp ark,t,scp:$data/mfcc_vad.ark,$data/mfcc_vad.scp
compute-fbank-feats --config=$conf/fbank.conf scp:$data/wav.scp ark,t,scp:$data/fbank_feats.ark,$data/fbank_feats.scp
compute-vad --config=$conf/vad.conf scp:$data/fbank_feats.scp ark,t,scp:$data/fbank_vad.ark,$data/fbank_vad.scp
nnet3-compute --use-gpu=no $nnet/final.last_hid.raw ark:$data/fbank_feats.ark ark,t,scp:$data/dvector.ark,$data/dvector.scp

