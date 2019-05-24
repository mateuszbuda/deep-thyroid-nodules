# Thyroid nodules


## Reproduction steps


### Docker

```
docker build -t deep-thyroid-nodules .
```

```
docker run --rm -it -v `pwd`:/workspace deep-thyroid-nodules
```

To access data folders from docker, use `-v` option.


### Prerequisites

1. Mount network drive with data and copy the folder with images `WashU/Nodules` to local drive

    ```
    sudo mkdir -p /media/maciej/thyroid/WashU/Nodules
    sudo mount -t cifs -o user=mb527 //duhsnas-pri.dhe.duke.edu/dusom_railabs/All_Staff/maciej-collaborators/thyroid /media/maciej/thyroid/
    cp -r /media/maciej/thyroid/WashU/Nodules /media/maciej/Thyroid/thyroid-nodules/
    ```

    The copied folder should contain `2866` images.

2. Install Tensorflow object detection API as described in the installation section of
[object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)


### Data cleaning

#### Convert images to PNG

Use MatLab function `tif2png()` in `data-cleaning` folder to convert all images to PNG.

#### Curate file names

0. find the case with `call` in name and rename it to have `cal` instead
1. remove files without `cal` in name using ``` rm `ls . | grep -v cal` ```
2. rename file with `*[_trans|_long]*` to `*[.trans|.long]*`
3. rename files with `*_0[1|2|3]*` to `*_[1|2|3]*` using `rename -v -n 's/(\d)_0(\d{1})(.*$)/${1}_${2}${3}/' *`
4. rename files that are not 0 padded using `rename -v -n 's/^(\d{3})([._]{1})(.*$)/0${1}${2}${3}/' *`
5. remove all files that were not renamed in the previous point

> `-n` option in `rename` makes it print names of files to be renamed, but don't rename.

#### Corrupted images

Remove corrupted images (with some heat-map overlay):

- `rm 0261_2.long.cal.PNG`
- `rm 1281.long.cal.PNG`

At this point, the folder with images should contain `2838` files.

#### Unpaired images

First, copy a double-view image for transversal view only using `cp 0498.trans.cal.PNG 0498.long.cal.PNG`.

Then, run `python remove_unpaired_views.py` to remove unpaired images (with image for only one view).

At this point, the folder with images should contain `2830` files.

#### Double views

First, remove the case with one single and one double view and copy its double view:
`rm 0181.long.cal.PNG && cp 0181.trans.cal.PNG 0181.long.cal.PNG`.

Split images with double views using `python split_double_views.py` in `data-cleaning` folder.

#### Test split

Move 99 test cases to a separate folder.

    mkdir -p /media/maciej/Thyroid/thyroid-nodules/Nodules-test/
    python test_split.py

Now, `Nodules-test` should contain `198` images and `Nodules` should contain `2632` images.

#### Unlabeled cases

Remove images without labels (not present in `train_ids.csv`) using `python remove_unlabeled.py`.

Now, `Nodules` should contain `2628` images

#### Pre-processing for detection

Copy files that will be used and pre-processed for detection

    cp -rp /media/maciej/Thyroid/thyroid-nodules/Nodules /media/maciej/Thyroid/thyroid-nodules/detection/Nodules`

Run MatLab pre-processing function `preprocessing()` from `detection` folder.

#### Bounding box annotation for callipers

Annotate callipers using MatLab function `bbox.m` which takes two arguments `images_regex` and `bboxes_out`.

The `images_regex` input variable in `bbox` function is a regex path to the directory with US images for annotation.

`*.csv` files with recorded coordinates are saved to `bboxes_out` directory, which must be created before the script is called.
Naming convention of saved `*.csv` files is the same as a corresponding image file name with extension changed to `*.csv`.

Format of `*.csv` files is that it contains 2 columns.
The first one is for row and the other for column pixel.
Rows correspond to 2-dimensional points.

E.g.

    12,34
    910,1112

The file above represents three points with MatLab convention coordinates (row, column) `(12, 34)`, and `(910, 1112)`.

If no points were recorded for an image, the corresponding `*.csv` will be empty.

#### Invalid calipers

Keep only the cases with valid numbers of callipers, i.e. `(2, 2)`, `(2, 4)`, and `(4, 4)`.

Remove invalid cases using `python remove_unpaired_cal.py`.

At this point, the folder with images should contain `2556` files.


### Detection

#### TF object detection set-up

Folder structure in root folder `/media/maciej/Thyroid/thyroid-nodules`:

    + detection
        + [0-9]  # fold number
            + data
            + model
                + eval
                + images
                + train
                - faster_rcnn_resnet101_coco.config
        + fine_tune_ckpt
            - model.ckpt.data-00000-of-00001
            - model.ckpt.index
            - model.ckpt.meta
        - label_map.pbtxt

The config file `faster_rcnn_resnet101_coco.config` for fold 0 and the label mapping file `label_map.pbtxt` are provided in the `detection` folder.

`faster_rcnn_resnet101_coco` model checkpoint for initialization can be downloaded from
[detection_model_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

#### Cross-validation train/valid tfrecord data

Assuming folder structure set-up as in the previous step, generate training data for detection using `csv2tfrecords.py` script in the `detection` folder:

    for i in `seq 0 9`; do python csv2tfrecords.py $i; done

#### Training

Change directory to `~/models/research/object_detection` and run for all folds:

    python train.py --gpu 1 \
        --pipeline_config_path /media/maciej/Thyroid/thyroid-nodules/detection/0/model/faster_rcnn_resnet101_coco.config \
        --train_dir /media/maciej/Thyroid/thyroid-nodules/detection/0/model/train/

To follow the performance on a validation set, run:

    python eval.py --gpu 0 \
        --checkpoint_dir /media/maciej/Thyroid/thyroid-nodules/detection/0/model/train/ \
        --eval_dir /media/maciej/Thyroid/thyroid-nodules/detection/0/model/eval/ \
        --pipeline_config_path /media/maciej/Thyroid/thyroid-nodules/detection/0/model/faster_rcnn_resnet101_coco.config

#### Exporting models for inference

    for i in `seq 0 9`; do python export_inference_graph.py --input_type image_tensor \
        --pipeline_config_path /media/maciej/Thyroid/thyroid-nodules/detection/$i/model/faster_rcnn_resnet101_coco.config \
        --trained_checkpoint_prefix /media/maciej/Thyroid/thyroid-nodules/detection/$i/model/train/model.ckpt-20000 \
        --output_directory /media/maciej/Thyroid/thyroid-nodules/detection/$i/model/inference/ ; done

#### Inference

From `detection` folder run:

    mkdir -p /media/maciej/Thyroid/thyroid-nodules/detection/Nodules-cv-bboxes
    mkdir -p /media/maciej/Thyroid/thyroid-nodules/detection/Calipers-cv
    python inference.py

In folder `Nodules-cv-bboxes` there are images with bounding boxes overlaid,
whereas `Calipers-cv` folder contains csv files with coordinates of bounding boxes for corresponding images.

#### Evaluation

To compute precision@\[.5:.95\]IoU for a nodule level detection, run evaluation script

    python evaluation.py

#### Postprocessing

From `detection` folder run:

    python postprocessing.py '/media/maciej/Thyroid/thyroid-nodules/detection/Calipers-cv/*.csv'


### Detection baseline

This section covers only the training set and cross-validation experiment.

#### TF object detection set-up

Folder structure in root folder `/media/maciej/Thyroid/thyroid-nodules`:

    + detection-baseline
        + [0-9]  # fold number
            + data
            + model
                + eval
                + images
                + train
                - faster_rcnn_resnet101_coco.config
        + fine_tune_ckpt
            - model.ckpt.data-00000-of-00001
            - model.ckpt.index
            - model.ckpt.meta
        - label_map.pbtxt

The config file `faster_rcnn_resnet101_coco.config` for fold 0 and the label mapping file `label_map.pbtxt` are provided in the `detection-baseline` folder.

`faster_rcnn_resnet101_coco` model checkpoint for initialization can be downloaded from
[detection_model_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

#### Cross-validation train/valid tfrecord data

Assuming folder structure set-up as in the previous step, generate training data for detection using `csv2tfrecords.py` script in the `detection-baseline` folder:

    for i in `seq 0 9`; do python csv2tfrecords.py $i; done

#### Training

Change directory to `~/models/research/object_detection` and run for all folds:

    python train.py --gpu 1 \
        --pipeline_config_path /media/maciej/Thyroid/thyroid-nodules/detection-baseline/0/model/faster_rcnn_resnet101_coco.config \
        --train_dir /media/maciej/Thyroid/thyroid-nodules/detection-baseline/0/model/train/

To follow the performance on a validation set, run:

    python eval.py --gpu 0 \
        --checkpoint_dir /media/maciej/Thyroid/thyroid-nodules/detection-baseline/0/model/train/ \
        --eval_dir /media/maciej/Thyroid/thyroid-nodules/detection-baseline/0/model/eval/ \
        --pipeline_config_path /media/maciej/Thyroid/thyroid-nodules/detection-baseline/0/model/faster_rcnn_resnet101_coco.config

#### Exporting models for inference

    for i in `seq 0 9`; do python export_inference_graph.py --input_type image_tensor \
        --pipeline_config_path /media/maciej/Thyroid/thyroid-nodules/detection-baseline/$i/model/faster_rcnn_resnet101_coco.config \
        --trained_checkpoint_prefix /media/maciej/Thyroid/thyroid-nodules/detection-baseline/$i/model/train/model.ckpt-20000 \
        --output_directory /media/maciej/Thyroid/thyroid-nodules/detection-baseline/$i/model/inference/ ; done

#### Inference

From `detection-baseline` folder run:

    mkdir -p /media/maciej/Thyroid/thyroid-nodules/detection-baseline/Nodules-cv-bboxes
    mkdir -p /media/maciej/Thyroid/thyroid-nodules/detection-baseline/Calipers-cv
    python inference.py

In folder `Nodules-cv-bboxes` there are images with bounding boxes overlaid,
whereas `Calipers-cv` folder contains csv files with coordinates of bounding boxes for corresponding images.

To run inference with different post-processing, set the `POSTPROCESSING` variable in `inference.py`.

#### Evaluation

To compute precision@\[.5:.95\]IoU for a nodule level detection, run evaluation script

    python evaluation.py


### Malignancy training

#### Prepare images

Generating images for training based on postprocessed calipers.

First, copy the original images since postprocessing is done in place.

    cp -rp /media/maciej/Thyroid/thyroid-nodules/Nodules /media/maciej/Thyroid/thyroid-nodules/images-cv

From `traintools` folder run MatLab function:

    crop_images('/media/maciej/Thyroid/thyroid-nodules/images-cv/*.PNG', '/media/maciej/Thyroid/thyroid-nodules/detection/Calipers-cv')

#### Training

    mkdir -p /media/maciej/Thyroid/thyroid-nodules/multitask/custom/logs
    mkdir -p /media/maciej/Thyroid/thyroid-nodules/multitask/custom/checkpoints
    gpu=1
    fold=0
    python train_cv.py $gpu $fold

#### Testing

Generating predictions for test and training folds:

    python test_cv.py

Output file for cases in test folds are saved to `results/data/predictions_cv.csv` with columns \[`ID`, `Prediction`, `Cancer`, `Fold`\].
File `results/data/predictions_cv_train.csv` has the same format but contains prediction for training folds (each example appears 9 times).

## Reproduction steps for 99 test cases

Assumes that the previous section is completed.

### Detection

#### Pre-processing for detection

Copy files that will be used and pre-processed for detection

    cp -rp /media/maciej/Thyroid/thyroid-nodules/Nodules-test /media/maciej/Thyroid/thyroid-nodules/detection/Nodules-test

Run MatLab pre-processing function `preprocessing('/media/maciej/Thyroid/thyroid-nodules/detection/Nodules-test/*.PNG')` from `detection` folder.

#### TF object detection set-up

Folder structure in root folder `media/maciej/Thyroid/thyroid-nodules`:

    + detection
        + test
            + data
            + model
                + eval
                + images
                + train
                - faster_rcnn_resnet101_coco.config
        + fine_tune_ckpt
            - model.ckpt.data-00000-of-00001
            - model.ckpt.index
            - model.ckpt.meta
        - label_map.pbtxt

The config file `faster_rcnn_resnet101_coco.config` for fold `0` and the label mapping file `label_map.pbtxt` are provided in the `detection` folder.

`faster_rcnn_resnet101_coco` model checkpoint for initialization can be downloaded from
[detection_model_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

#### Training and test tfrecord data

Assuming folder structure set-up as in the previous step, generate training data for detection using `csv2tfrecords_99test.py` script in the `detection` folder:

    python csv2tfrecords_99test.py

#### Training

Change directory to `~/models/research/object_detection` and run for all folds:

    python train.py --gpu 1 \
        --pipeline_config_path /media/maciej/Thyroid/thyroid-nodules/detection/test/model/faster_rcnn_resnet101_coco.config \
        --train_dir /media/maciej/Thyroid/thyroid-nodules/detection/test/model/train/

To follow the performance on a validation set, run:

    python eval.py --gpu 0 \
        --checkpoint_dir /media/maciej/Thyroid/thyroid-nodules/detection/test/model/train/ \
        --eval_dir /media/maciej/Thyroid/thyroid-nodules/detection/test/model/eval/ \
        --pipeline_config_path /media/maciej/Thyroid/thyroid-nodules/detection/test/model/faster_rcnn_resnet101_coco.config

#### Exporting models for inference

    python export_inference_graph.py --input_type image_tensor \
        --pipeline_config_path /media/maciej/Thyroid/thyroid-nodules/detection/test/model/faster_rcnn_resnet101_coco.config \
        --trained_checkpoint_prefix /media/maciej/Thyroid/thyroid-nodules/detection/test/model/train/model.ckpt-20000 \
        --output_directory /media/maciej/Thyroid/thyroid-nodules/detection/test/model/inference/

#### Inference

From `detection` folder run:

    mkdir -p /media/maciej/Thyroid/thyroid-nodules/detection/Nodules-test-bboxes
    mkdir -p /media/maciej/Thyroid/thyroid-nodules/detection/Calipers-test
    python inference_99test.py

In folder `Nodules-test-bboxes` there are images with bounding boxes overlaid,
whereas `Calipers-test` folder contains csv files with coordinates of bounding boxes for corresponding images.

#### Postprocessing

From `detection` folder run:

    python postprocessing.py '/media/maciej/Thyroid/thyroid-nodules/detection/Calipers-test/*.csv'


### Malignancy training

#### Prepare images

Generating images for training based on postprocessed calipers.

First, copy the original images since postprocessing is done in place.

    cp -rp /media/maciej/Thyroid/thyroid-nodules/Nodules-test /media/maciej/Thyroid/thyroid-nodules/images-test

From `traintools` folder run MatLab function:

    crop_images('/media/maciej/Thyroid/thyroid-nodules/images-test/*.PNG', '/media/maciej/Thyroid/thyroid-nodules/detection/Calipers-test')

#### Training

    mkdir -p /media/maciej/Thyroid/thyroid-nodules/multitask/custom-test/logs
    mkdir -p /media/maciej/Thyroid/thyroid-nodules/multitask/custom-test/checkpoints
    gpu=1
    python train_99test.py $gpu

#### Testing

Generating predictions for test folds:

    python test_99test.py

And for training cases:

    python test_train.py

Training predictions are used to compute thresholds for deep learning risk levels corresponding to TI-RADS risk levels

Output file for cases in the test set are saved to `results/data/predictions_test.csv` with columns \[`ID`, `Prediction`, `Cancer`\].
File `results/data/predictions_train.csv` has the same format but contains prediction for the training cases.

## Results analysis and comparison to radiologists

All notebooks that read predictions from deep learning models and compare them to radiologists are in the `results` folder.

    cd results
    jupyter notebook
