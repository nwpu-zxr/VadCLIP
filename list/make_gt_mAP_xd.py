import numpy as np
import glob
import os
import cv2
import pandas as pd
import warnings

clip_len = 16

# the dir of testing images
feature_list = 'list/xd_CLIP_rgbtest.csv'

# the ground truth txt
gt_txt = 'list/annotations_multiclasses.txt'
gt_lines = list(open(gt_txt))

#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

gt_segment = []
gt_label = []
lists = pd.read_csv(feature_list)

for idx in range(lists.shape[0]):
    name = lists.loc[idx]['path']
    if '__0.npy' not in name:
        continue
    segment = []
    label = []
    if '_label_A' in name:
        fea = np.load(name)
        lens = fea.shape[0] * clip_len
        name = name.split('/')[-1]
        name = name[:-7]
        segment.append([0, lens])
        label.append('A')
    else:
        name = name.split('/')[-1]
        name = name[:-7]
        for gt_line in gt_lines:
            if name in gt_line:
                gt_content = gt_line.strip('\n').split()
                for j in range(1, len(gt_content), 3):
                    print(gt_content, j)
                    segment.append([gt_content[j + 1], gt_content[j + 2]])
                    label.append(gt_content[j])
                break
    gt_segment.append(segment)
    gt_label.append(label)
    
np.save('list/gt_label.npy', gt_label)
np.save('list/gt_segment.npy', gt_segment)