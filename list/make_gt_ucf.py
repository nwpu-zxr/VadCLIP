import numpy as np
import pandas as pd
import cv2

clip_len = 16

# the dir of testing images
feature_list = 'list/ucf_CLIP_rgbtest.csv'
# the ground truth txt

gt_txt = 'list/Temporal_Anomaly_Annotation.txt'     ## the path of test annotations
gt_lines = list(open(gt_txt))
gt = []
lists = pd.read_csv(feature_list)
count = 0

for idx in range(lists.shape[0]):
    name = lists.loc[idx]['path']
    if '__0.npy' not in name:
        continue
    #feature = name.split('label_')[-1]
    fea = np.load(name)
    lens = (fea.shape[0] + 1) * clip_len
    name = name.split('/')[-1]
    name = name[:-7]
    # the number of testing images in this sub-dir

    gt_vec = np.zeros(lens).astype(np.float32)
    if 'Normal' not in name:
        for gt_line in gt_lines:
            if name in gt_line:
                count += 1
                gt_content = gt_line.strip('\n').split('  ')[1:-1]
                abnormal_fragment = [[int(gt_content[i]),int(gt_content[j])] for i in range(1,len(gt_content),2) \
                                        for j in range(2,len(gt_content),2) if j==i+1]
                if len(abnormal_fragment) != 0:
                    abnormal_fragment = np.array(abnormal_fragment)
                    for frag in abnormal_fragment:
                        if frag[0] != -1 and frag[1] != -1:
                            gt_vec[frag[0]:frag[1]]=1.0
                break
    gt.extend(gt_vec[:-clip_len])

print(count)
np.save('list/gt_ucf.npy', gt)