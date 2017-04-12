## d-vector speaker segmentation in THU emotion dataset

No. conversations: 1,601

No. change points: 32,992

Tolerance: 0.3 seconds

### Sliding threshold:
FAR and MDR evolution of three tolerances:
![FAR and MDR](result/thres_seg/FAR_and_MDR.png)


### local minimum detection:

- detected points: 233,472
- false alarm: 185,074
- miss detection: 5,428
- FAR: 83.85%
- MDR: 16.62%

### k-means clustering:
- detected points: 42,148
- false alarm: 19,468
- miss detection: 11,177
- FAR: 31.59% (var: 3.295)
- MDR: 33.97% (var: 3.327)

The distribution of the two error rate in the dataset
![distribution1](result/kmeans_cluster/FAR_side.png)
![distribution2](result/kmeans_cluster/MDR_side.png)

### known two speakers -5 segments to train model:
- detected points: 40,849
- false alarm: 17,622
- miss detection: 10,341
- FAR: 31.70% (var: 2.031)
- MDR: 31.48% (var: 2.772)

The distribution of the two error rate in the dataset
![distribution1](result/know_2_seg/know2spk_5_1.png)
![distribution2](result/know_2_seg/know2spk_5_2.png)

### known two speakers -all segments to train model:
- detected points: 39,011
- false alarm: 14,988
- miss detection: 9,473
- FAR: 28.46% (var: 1.950)
- MDR: 28.84% (var: 2.605)

The distribution of the two error rate in the dataset
![distribution1](result/know_2_seg/know2spk_all_1.png)
![distribution2](result/know_2_seg/know2spk_all_2.png)


## Design of software

### feature extraction: d-vector (with kaldi toolkit)

input: wavfile, nnet, wavlist

output: d-vector feature files (with pca feature dimensionality reduction)

### bic, glr, kl2 segmentation

input: mfcc sequences (numpy array), window_size, window_shift

output: score curves, initial segmentation blocks(start_time, end_time)

### d-vector segmentation

input: d-vector sequences (numpy array), window_size, window_shift

output: score curves, initial segmentation blocks(start_time, end_time)

### generate segmentation blocks with fixed_threshold

input: score curves (bic, glr, kl2, d-vector)

output: new segmentation blocks

### blind resegmentation k-means clustering (default k=2)

input: initial segmentation blocks

output: new segmentation blocks, speaker tag

### resegmentation with speaker models

input: initial segmentation blocks, speaker models

output: new segmentation blocks, speaker tag

### evalution

FAR, MDR: segmentation blocks, ref file, tolerance

DET curves with sliding threshold: d-vector score curves(list), ref file(list)

ASP, ACP: segmentation blocks, speaker tag, ref file(with speaker tag)

speaker1_false_rate, speaker1_miss_rate, speaker2_false_rate, speaker2_miss_rate: segmentation blocks, speaker tag, ref file(with speaker tag). - divided by speaker1 and speaker2


### warning:
- single converation segmentation and list segmentation configuration, especially in sliding threshold evluation

