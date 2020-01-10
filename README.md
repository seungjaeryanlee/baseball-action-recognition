# baseball-action-recognition

## Prerequisites

This repository was tested in Ubuntu 18.04 LTS with Python 3.7.4.

The Python requirements are written in `requirements.txt`. Additionally, to download the Baseball Database (BBDB), you need `aria2`.

## Prepare the Dataset

This repository has two script files to

1. Download full videos from BBDB, and
2. Extract segments from full videos

Run the two lines below in order:

```bash
python scripts/download_bbdb.py -i bbdb.selected.v0.9.min.json
python scripts/extract_segments_from_videos.py
```

This should download 45 videos and extract 14826 segments. The segments are divided into train-valid-test split with 60% (8896), 20% (2965), 20% (2965) each. The exact split can be seen at `data_split.min.json`.

## Train I3D

The I3D has two streams: RGB and Flow. Currently this repository only contains code for training the RGB stream. Check the `CONFIG` dictionary for the hyperparameters.

```
python train.py
```

## Evaluate I3D

After the training has completed, you can evaluate the model by specifying the model with `CONFIG["RGB_I3D_LOAD_MODEL_PATH"]` or `CONFIG["FLOW_I3D_LOAD_MODEL_PATH"]`. Check the `CONFIG` dictionary for additional hyperparameters.

```
python evaluate.py
```
