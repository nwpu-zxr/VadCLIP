import os
import glob
import csv

root_path = '/home/xbgydx/Desktop/XDTrainClipFeatures'    ## the path of features
files = sorted(glob.glob(os.path.join(root_path, "*.npy")))
violents = []
normal = []

with open('list/xd_CLIP_rgb.csv', 'w+') as f:  ## the name of feature list
    writer = csv.writer(f)
    writer.writerow(['path', 'label'])
    for file in files:
        if '.npy' in file:
            if '_label_A' in file:
                normal.append(file)
            else:
                label = file.split('_label_')[1].split('__')[0]
                writer.writerow([file, label])
            
    for file in normal:
        writer.writerow([file, 'A'])