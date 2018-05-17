import numpy as np
import os

from glob import glob

PRED_CSV = '/media/maciej/Thyroid/thyroid-nodules/detection-baseline/Calipers-cv'
GT_CSV_RGX = '/media/maciej/Thyroid/thyroid-nodules/detection/Calipers/*csv'

def get_gt_bbox(calipers_path):
	xmins = []
	ymins = []
	xmaxs = []
	ymaxs = []
	
	for line in open(calipers_path):
		values = line.rstrip().split(',')
		x = float(values[1])
		y = float(values[0])
		xmins.append(x - 32)
		ymins.append(y - 32)
		xmaxs.append(x + 32)
		ymaxs.append(y + 32)
	
	if len(xmins) == 2:
		ymid = (min(ymins) + max(ymaxs)) / 2
		w = max(xmaxs) - min(xmins)
		ymins.append(ymid - (w / 2))
		ymaxs.append(ymid + (w / 2))
		xmid = (min(xmins) + max(xmaxs)) / 2
		h = max(ymaxs) - min(ymins)
		xmins.append(xmid - (h / 2))
		xmaxs.append(xmid + (h / 2))
	
	xmins = np.maximum(xmins, 0.0)
	ymins = np.maximum(ymins, 0.0)
	
	xmin = min(xmins)
	ymin = min(ymins)
	xmax = max(xmaxs)
	ymax = max(ymaxs)
	
	return [ymin, xmin, ymax, xmax]
	

def get_pred_bboxes(bbox_path):
	# predicted bbouxes are in format [ymin, ymax, xmin, xmax]
	bboxes = []
	for line in open(bbox_path):
		values = line.rstrip().split(',')
		values = [float(x) for x in values]
		bboxes.append([values[0], values[2], values[1], values[3]])
	
	return bboxes


def bboxes_iou(boxA, boxB):
	# box is in format [ymin, xmin, ymax, xmax]
	# determine the (x, y)-coordinates of the intersection rectangle
	yA = max(boxA[0], boxB[0])
	xA = max(boxA[1], boxB[1])
	yB = min(boxA[2], boxB[2])
	xB = min(boxA[3], boxB[3])
	
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
	
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	
	# return the intersection over union value
	return iou


files = glob(GT_CSV_RGX)

precisions = []
hits = []
misses = []
for iou in np.linspace(0.5, 0.95, 10):
	hit_count = 0.
	miss_count = 0.
	for csv_path in files:
		gt_bbox = get_gt_bbox(csv_path)
		pred_bbox_path = os.path.join(PRED_CSV, os.path.split(csv_path)[1])
		pred_bboxes = get_pred_bboxes(pred_bbox_path)
		for pred_bbox in pred_bboxes:
			if bboxes_iou(gt_bbox, pred_bbox) < iou:
				miss_count += 1
				continue
			hit_count += 1.
	precisions.append(np.round(hit_count / (hit_count + miss_count) * 100, 2))
	hits.append(hit_count)
	misses.append(miss_count)

print('Precision@0.5IoU = {}'.format(precisions[0]))
print('Mean Precision@[.5:.95]IoU = {}'.format(np.mean(precisions)))
print(precisions)
print(hits)
print(misses)