import csv
import os
from glob import glob
from random import randint, seed

import numpy as np
import png
import tensorflow as tf
from PIL import Image
from medpy.filter.binary import largest_connected_component
from scipy.ndimage.morphology import binary_dilation

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# This is needed since the notebook is stored in the object_detection folder.
from object_detection.utils import ops as utils_ops

if tf.__version__ < "1.4.0":
    raise ImportError("Please upgrade your tensorflow installation to v1.4.* or later!")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from evaluation import get_gt_bbox

PATH_TO_LABELS = "./label_map.pbtxt"
NUM_CLASSES = 1
PATH_TO_SAVE_IMG = (
    "/media/maciej/Thyroid/thyroid-nodules/detection-baseline/Nodules-cv-bboxes"
)
PATH_TO_SAVE_CSV = (
    "/media/maciej/Thyroid/thyroid-nodules/detection-baseline/Calipers-cv"
)

# Post-processing strategies
ONE_BBOX = 101
NO_OVERLAP_ABOVE_TH = 102
ALL_ABOVE_TH = 103

POSTPROCESSING = ONE_BBOX

min_score_thresh = 0.5
overlay_bboxes = True


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def numpy2png(image, path):
    with open(path, "wb") as pngfile:
        pngWriter = png.Writer(
            image.shape[1], image.shape[0], greyscale=False, alpha=False, bitdepth=8
        )
        pngWriter.write(pngfile, np.reshape(image, (-1, np.prod(image.shape[1:]))))


def fname2pid(fname):
    return fname.split("/")[-1].split(".")[0].lstrip("0")


def run_inference_for_single_image(image, sess):
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        "num_detections",
        "detection_boxes",
        "detection_scores",
        "detection_classes",
        "detection_masks",
    ]:
        tensor_name = key + ":0"
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    if "detection_masks" in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict["detection_boxes"], [0])
        detection_masks = tf.squeeze(tensor_dict["detection_masks"], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict["num_detections"][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(
            detection_masks, [0, 0, 0], [real_num_detection, -1, -1]
        )
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1]
        )
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.0), tf.uint8
        )
        # Follow the convention by adding back the batch dimension
        tensor_dict["detection_masks"] = tf.expand_dims(detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")

    # Run inference
    output_dict = sess.run(
        tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)}
    )

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict["num_detections"] = int(output_dict["num_detections"][0])
    output_dict["detection_classes"] = output_dict["detection_classes"][0].astype(
        np.uint8
    )
    output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
    output_dict["detection_scores"] = output_dict["detection_scores"][0]
    if "detection_masks" in output_dict:
        output_dict["detection_masks"] = output_dict["detection_masks"][0]
    return output_dict


for f in range(10):
    FOLD = f
    PATH_TO_CKPT = "/media/maciej/Thyroid/thyroid-nodules/detection-baseline/{}/model/inference/frozen_inference_graph.pb".format(
        FOLD
    )

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)

    TEST_IMAGE_PATHS = []

    image_files = glob("/media/maciej/Thyroid/thyroid-nodules/detection/Nodules/*.PNG")
    pid_set = set()
    for image_f in image_files:
        pid = fname2pid(image_f)
        pid_set.add(pid)

    seed(42)
    test_folds = {}  # maps pid to its test fold number in 10-fold cross-validation split
    for pid in sorted(pid_set):
        test_folds[pid] = randint(0, 9)

    for image_f in image_files:
        pid = fname2pid(image_f)
        if test_folds[pid] == FOLD:
            TEST_IMAGE_PATHS.append(image_f)

    with detection_graph.as_default():
        with tf.Session() as sess:
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                x_pad_size = 0
                if image_np.shape[0] - image_np.shape[1] > 200:
                    print("padding {}".format(image_path))
                    x_pad_size = int(0.1 * image_np.shape[1])
                    image_np = np.pad(
                        image_np, ((0, 0), (x_pad_size, x_pad_size), (0, 0)), "constant"
                    )
                # Actual detection.
                output_dict = run_inference_for_single_image(image_np, sess)

                # get bounding box for the ROI
                image_gs = np.mean(image_np, axis=-1)
                image_bw = np.greater(image_gs, 3.0)
                image_bw = binary_dilation(
                    image_bw, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
                )
                image_th = largest_connected_component(image_bw)
                cc_indices = np.argwhere(image_th)
                y_min_th = float(np.min(cc_indices[:, 0])) / image_np.shape[0]
                y_max_th = float(np.max(cc_indices[:, 0])) / image_np.shape[0]
                x_min_th = float(np.min(cc_indices[:, 1])) / image_np.shape[1]
                x_max_th = float(np.max(cc_indices[:, 1])) / image_np.shape[1]

                # filter the result
                detection_boxes = []
                detection_scores = []
                detection_classes = []
                for i in range(output_dict["num_detections"]):
                    score = output_dict["detection_scores"][i]
                    # filter for score
                    if score > min_score_thresh:
                        # and filter for points outside of the ROI
                        bbox = output_dict["detection_boxes"][i]
                        y = (bbox[0] + bbox[2]) / 2
                        x = (bbox[1] + bbox[3]) / 2
                        if y < y_min_th or y > y_max_th or x < x_min_th or x > x_max_th:
                            continue
                        if POSTPROCESSING == NO_OVERLAP_ABOVE_TH:
                            # and filter for overlapping points (midpoint inside already included bbox)
                            if any(
                                [
                                    (db[0] <= y <= db[2] and db[1] <= x <= db[3])
                                    for db in detection_boxes
                                ]
                            ):
                                continue
                        detection_boxes.append(output_dict["detection_boxes"][i])
                        detection_scores.append(output_dict["detection_scores"][i])
                        detection_classes.append(output_dict["detection_classes"][i])

                if len(detection_boxes) < 1:
                    print("below 0.5 threshold {}".format(image_path))

                det_index = -1
                while len(detection_boxes) < 1:
                    det_index += 1
                    if det_index >= len(output_dict["detection_boxes"]):
                        break
                    # again filter for points outside of the ROI
                    bbox = output_dict["detection_boxes"][det_index]
                    y = (bbox[0] + bbox[2]) / 2
                    x = (bbox[1] + bbox[3]) / 2
                    if y < y_min_th or y > y_max_th or x < x_min_th or x > x_max_th:
                        continue
                    if POSTPROCESSING == NO_OVERLAP_ABOVE_TH:
                        # and again filter for overlapping points (midpoint inside already included bbox)
                        if any(
                            [
                                (db[0] <= y <= db[2] and db[1] <= x <= db[3])
                                for db in detection_boxes
                            ]
                        ):
                            continue

                    detection_boxes.append(output_dict["detection_boxes"][det_index])
                    detection_scores.append(output_dict["detection_scores"][det_index])
                    detection_classes.append(
                        output_dict["detection_classes"][det_index]
                    )

                if POSTPROCESSING == ONE_BBOX:
                    detection_boxes = detection_boxes[:1]
                    detection_scores = detection_scores[:1]
                    detection_classes = detection_classes[:1]

                csv_result_path = os.path.join(
                    PATH_TO_SAVE_CSV, os.path.split(image_path)[-1]
                )
                csv_result_path = csv_result_path.replace("PNG", "csv")

                with open(csv_result_path, "a") as f:
                    writer = csv.writer(f)
                    for i in range(len(detection_boxes)):
                        bbox = detection_boxes[i]
                        ymin = int(bbox[0] * image_np.shape[0])
                        xmin = max(int(bbox[1] * image_np.shape[1]) - x_pad_size, 0)
                        ymax = int(bbox[2] * image_np.shape[0])
                        xmax = min(
                            int(bbox[3] * image_np.shape[1]) - x_pad_size,
                            image_np.shape[1] - 1,
                        )
                        writer.writerow([ymin, ymax, xmin, xmax, detection_scores[i]])

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.array(detection_boxes),
                    np.array(detection_classes),
                    np.array(detection_scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4,
                    min_score_thresh=0.0,
                )

                image_np = image_np[:, x_pad_size : -x_pad_size - 1, :]
                gt_box = get_gt_bbox(
                    image_path.replace("Nodules", "Calipers").replace("PNG", "csv")
                )

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.array([gt_box]),
                    np.array([1]),
                    None,
                    category_index,
                    use_normalized_coordinates=False,
                    line_thickness=4,
                )

                img_result_path = os.path.join(
                    PATH_TO_SAVE_IMG, os.path.split(image_path)[-1]
                )
                numpy2png(image_np, img_result_path)
