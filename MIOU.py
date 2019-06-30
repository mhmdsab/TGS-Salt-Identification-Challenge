import numpy as np

def MIOU(logits, labels, threshold, batch_size):
    ious = []
    for i in range(batch_size):
        Nlabels, Nlogits = labels[i, :, :].copy(), logits[i, :, :].copy()
        Nlogits[Nlogits > threshold]= 1
        Nlogits[Nlogits < threshold]= 0
        intersection = np.sum(Nlogits.astype(np.int32) & Nlabels.astype(np.int32))
        union = np.sum(Nlogits.astype(np.int32) | Nlabels.astype(np.int32))
        if intersection == 0 and union == 0: iou = 1
        else: iou = intersection/(union+1e-8)
        ious.append(iou)
    return ious
#a = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0])
#a = a.reshape([2, 5, 5])
#b = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1])
#b = b.reshape([2, 5, 5])
#
#c = MIOU(a, b, 0.5, 2)
#print(a[0, :, :])
#print(b[0, :, :])
#print(c[0])

def Kaggle_MIOU(logits, labels, thresholds, batch_size):
    per_batch = []
    for i in range(batch_size):
        per_thresh = []
        for thresh in thresholds:
            Nlabels, Nlogits = labels[i, :, :].copy(), logits[i, :, :].copy()
            Nlogits[Nlogits >= thresh]= 1
            Nlogits[Nlogits < thresh]= 0
            intersection = np.sum(Nlogits.astype(np.int32) & Nlabels.astype(np.int32))
            union = np.sum(Nlogits.astype(np.int32) | Nlabels.astype(np.int32))
            if intersection == 0 and union == 0: iou = 1
            else: iou = intersection/(union+1e-8)
            per_thresh.append(iou)
        per_batch.append(np.mean(np.array(per_thresh)))
    return per_batch
