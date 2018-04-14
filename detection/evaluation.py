import numpy as np
import os

from glob import glob

PRED_CSV = '/media/maciej/Thyroid/thyroid-nodules/detection/Calipers-cv'
GT_CSV_RGX = '/media/maciej/Thyroid/thyroid-nodules/detection/Calipers/*csv'

def get_bbox(calipers_path):
	bbox_x = []
	bbox_y = []
	for line in open(calipers_path):
		values = line.rstrip().split(',')
		x = float(values[1])
		y = float(values[0])
		bbox_x.append(x)
		bbox_y.append(y)
	
	len_x = (max(bbox_x) - min(bbox_x)) / 2.
	mid_y = (max(bbox_x) + min(bbox_x)) / 2.
	len_y = (max(bbox_y) - min(bbox_y)) / 2.
	mid_x = (max(bbox_y) + min(bbox_y)) / 2.
	radius = max(len_x, len_y) + 32
	
	bb = [max(0, mid_y - radius), max(0, mid_x - radius), mid_y + radius, mid_x + radius]
	return bb


def bboxes_iou(boxA, boxB):
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
for iou in np.linspace(0.5, 0.95, 10):
	hit_count = 0.
	for csv_path in files:
		gt_bbox = get_bbox(csv_path)
		pred_bbox_path = os.path.join(PRED_CSV, os.path.split(csv_path)[1])
		pred_bbox = get_bbox(pred_bbox_path)
		if bboxes_iou(gt_bbox, pred_bbox) < iou:
			continue
		hit_count += 1.
	precisions.append(hit_count / len(files))

print('Precision@0.5IoU = {}'.format(precisions[0]))
print('Mean Precision@[.5:.95]IoU = {}'.format(np.mean(precisions)))
