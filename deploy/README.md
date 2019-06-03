# Deployment for detection and classification of thyroid nodules seen in ultrasound

## Detection

1. Download `frozen_inference_graph.pb` to `./` folder.

2. Run preprocessing in matlab using `preprocessing.m`

3. Build docker image for detection

```
docker build -t deep-thyroid-nodules-detection -f Dockerfile.detection .
```

4. Place images with `*.PNG` extension preprocessed in step 2. for inference in `/path/to/data/Nodules-test/`

5. Run inference and postprocessing

```
nvidia-docker run --rm -it -v /path/to/data/:/data/ deep-thyroid-nodules-detection
```

Predicted locations of calipers used in the next step are in `/data/Calipers-test/`

## Classification

TODO
