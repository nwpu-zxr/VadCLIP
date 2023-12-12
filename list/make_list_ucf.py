import os
import csv

root_path = '/home/xbgydx/Desktop/UCFClipFeatures/'
txt = 'list/Anomaly_Train.txt'
files = list(open(txt))
normal = []
count = 0

with open('list/ucf_CLIP_rgb.csv', 'w+') as f:  ## the name of feature list
    writer = csv.writer(f)
    writer.writerow(['path', 'label'])
    for file in files:
        filename = root_path + file[:-5] + '__0.npy'
        label = file.split('/')[0]
        if os.path.exists(filename):
            if 'Normal' in label:
                #continue
                filename = filename[:-5]
                for i in range(0, 10, 1):
                    normal.append(filename + str(i) + '.npy')
            else:
                filename = filename[:-5]
                for i in range(0, 10, 1):
                    writer.writerow([filename + str(i) + '.npy', label])
        else:
            count += 1
            print(filename)
            
    for file in normal:
        writer.writerow([file, 'Normal'])

print(count)