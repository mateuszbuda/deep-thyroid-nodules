# Deployment for detection and classification of thyroid nodules seen in ultrasound

## Detection

1. Download `frozen_inference_graph.pb` from box folder `weights` to `./` folder.

2. Run preprocessing in matlab using `preprocessing.m`

3. Build docker image for detection

```
docker build -t deep-thyroid-nodules-detection -f Dockerfile.detection .
```

4. Place images with `*.PNG` extension preprocessed in step 2. for inference in `/path/to/data/Nodules-test/`

5. Run inference and postprocessing

```
nvidia-docker run --rm -v /path/to/data/:/data/ deep-thyroid-nodules-detection
```

Predicted locations of calipers used in the next step are in `/data/Calipers-test/`

## Classification

1. Download `weights.h5` from box folder `weights` to `./` folder.

2. Run preprocessing in matlab using `crop_images.m`

3. Build docker image for classification

```
docker build -t deep-thyroid-nodules-classification -f Dockerfile.classification .
```

4. Place images with `*.PNG` extension preprocessed in step 2. for inference in `/path/to/data/images-test/`

5. Run testing

```
nvidia-docker run --rm -v /path/to/data/:/data/ deep-thyroid-nodules-classification
```

Predictions for malignancy are in `/data/predictions_test.csv`
