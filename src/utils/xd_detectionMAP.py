import numpy as np
from scipy.signal import savgol_filter

def smooth(v):
   return v
   # l = min(5, len(v)); l = l - (1-l%2)
   # if len(v) <= 3:
   #   return v
   # return savgol_filter(v, l, 1) #savgol_filter(v, l, 1) #0.5*(np.concatenate([v[1:],v[-1:]],axis=0) + v)

def str2ind(categoryname,classlist):
   return [i for i in range(len(classlist)) if categoryname == classlist[i]][0]

def nms(dets, thresh=0.6, top_k=-1):
    """Pure Python NMS baseline."""
    # dets: N*2 and sorted by scores
    if len(dets) == 0: return []
    order = np.arange(0,len(dets),1)
    dets = np.array(dets)
    x1 = dets[:, 0]  # start
    x2 = dets[:, 1]  # end
    lengths = x2 - x1 
    keep = []
    while order.size > 0:
        i = order[0] # the first is the best proposal
        keep.append(i) # put into the candidate pool
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]]) 
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) ## the intersection
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter) ## the iou
        inds = np.where(ovr <= thresh)[0]  # the index of remaining proposals
        order = order[inds + 1] # add 1

    return dets[keep], keep

def getLocMAP(predictions, th, gtsegments, gtlabels, excludeNormal):
   if excludeNormal is True:
       classes_num = 6
       videos_num = 500
       predictions = predictions[:videos_num]
   else:
       classes_num = 7
       videos_num = 800

   classlist = ['A', 'B1', 'B2', 'B4', 'B5', 'B6', 'G']
   predictions_mod = []
   c_score = []
   for p in predictions:
      pp = - p
      [pp[:,i].sort() for i in range(np.shape(pp)[1])]
      pp=-pp
      idx_temp = int(np.shape(pp)[0]/16)
      c_s = np.mean(pp[:idx_temp, :], axis=0)
      ind = c_s > 0.0
      c_score.append(c_s)
      predictions_mod.append(p*ind)
   predictions = predictions_mod
   ap = []
   for c in range(0, 7):
      segment_predict = []
      # Get list of all predictions for class c
      for i in range(len(predictions)):
         tmp = smooth(predictions[i][:, c])
         segment_predict_multithr = []
         thr_set = np.arange(0.6, 0.7, 0.1)
         for thr in thr_set:
            threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp))*thr  ###  0.8 is the best?
            vid_pred = np.concatenate([np.zeros(1), (tmp>threshold).astype('float32'), np.zeros(1)], axis=0)
            vid_pred_diff = [vid_pred[idt]-vid_pred[idt-1] for idt in range(1, len(vid_pred))]
            s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
            e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
            for j in range(len(s)):
               if e[j]-s[j]>=2:
                  segment_scores = np.max(tmp[s[j]:e[j]])+0.7*c_score[i][c]
                  segment_predict_multithr.append([i, s[j], e[j], segment_scores])               
                  # segment_predict.append([i, s[j], e[j], np.max(tmp[s[j]:e[j]])+0.7*c_score[i][c]])
         if len(segment_predict_multithr)!=0:
            segment_predict_multithr = np.array(segment_predict_multithr)
            segment_predict_multithr = segment_predict_multithr[np.argsort(-segment_predict_multithr[:,-1])]     
            _, keep = nms(segment_predict_multithr[:, 1:-1], 0.6)
            segment_predict.extend(list(segment_predict_multithr[keep]))
      segment_predict = np.array(segment_predict)

      # Sort the list of predictions for class c based on score
      if len(segment_predict) == 0:
         return 0
      segment_predict = segment_predict[np.argsort(-segment_predict[:,3])]

      # Create gt list 
      segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]] for i in range(len(gtsegments))
                    for j in range(len(gtsegments[i])) if str2ind(gtlabels[i][j], classlist) == c]
      gtpos = len(segment_gt)

      # Compare predictions and gt
      tp, fp = [], []
      for i in range(len(segment_predict)):
         flag = 0.
         best_iou = 0.0
         for j in range(len(segment_gt)):
            if segment_predict[i][0]==segment_gt[j][0]:
               gt = range(int(segment_gt[j][1]), int(segment_gt[j][2]))
               p = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
               IoU = float(len(set(gt).intersection(set(p))))/float(len(set(gt).union(set(p))))
               if IoU >= th:
                  flag = 1.
                  if IoU > best_iou:
                     best_iou = IoU
                     best_j = j
         if flag > 0:
            del segment_gt[best_j]
         tp.append(flag)
         fp.append(1.-flag)
      tp_c = np.cumsum(tp)
      fp_c = np.cumsum(fp)
      if sum(tp)==0:
         prc = 0.
      else:
         prc = np.sum((tp_c/(fp_c+tp_c))*tp)/gtpos
      ap.append(prc)
      # print(np.round(prc, 4))
   return 100*np.mean(ap)
  

def getDetectionMAP(predictions, segments, labels, excludeNormal=False):
   iou_list = [0.1, 0.2, 0.3, 0.4, 0.5]
   # iou_list = [0.5]
   dmap_list = []
   for iou in iou_list:
      # print('Testing for IoU {:.1f}'.format(iou))
      dmap_list.append(getLocMAP(predictions, iou, segments, labels, excludeNormal))
   return dmap_list, iou_list

